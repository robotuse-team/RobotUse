"""Post-episode sensor-only pickup/hold check; never a controller/collision gate.

The visual judge is a fresh independent request, NOT simulation ground truth.
No connector object poses, contacts or benchmark object metadata are read.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.request
from PIL import Image

from src.tools.gripper.evidence import classify_gripper_fraction


def capture_first_grasp_baseline(*, connector=None):
    return {"kind":"sensor_only_no_oracle", "started_monotonic_s":time.monotonic()}


def _stage(evidence, name):
    observations = evidence.get("observations") or {}
    stage = (observations.get(name) if isinstance(observations, dict) else None) or evidence.get(name)
    return stage if isinstance(stage, dict) else {}


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _z(stage):
    pose=stage.get("robot_ee_pose",stage.get("ee_pose",{}))
    if isinstance(pose,list) and len(pose)==4 and all(isinstance(row,list) and len(row)==4 for row in pose):
        value=pose[2][3]
    else:
        p=pose.get("position",{}) if isinstance(pose,dict) else {}
        value=p.get("z") if isinstance(p,dict) else None
    return value if _finite(value) else None


def _clock(stage):
    for key in ("simulation_time_s","sim_time_s"):
        if _finite(stage.get(key)):
            return stage[key]
    return None


def _recorded_lift(output):
    """Fallback: sampled actual EE just before lift versus last lift frame."""
    try:
        root=Path(output)/"interface"
        plans=[json.loads(x) for x in (root/"plans.jsonl").read_text().splitlines()]
        frames=[json.loads(x) for x in (root/"frames.jsonl").read_text().splitlines()]
        plan=next(p for p in reversed(plans) if p["kind"]=="grasp")
        labels=plan["target_labels"]
        index=labels.index("lift")
        sid=plan["segments"][index]["segment_id"]
        selected=[(i,f) for i,f in enumerate(frames) if f.get("segment_id")==sid]
        if not selected or selected[0][0]==0:
            return None
        before=frames[selected[0][0]-1]
        after=selected[-1][1]
        value=after["actual_ee_xyz"][2]-before["actual_ee_xyz"][2]
        return float(value) if _finite(value) else None
    except (OSError,KeyError,ValueError,TypeError,StopIteration):
        return None


def _visual_request(stages, *, model, api_key, objective, audit_path=None, provider_config=None):
    from src.llm.config import ProviderConfig
    config = provider_config or ProviderConfig.resolve(model)
    blocks=[{"type":"text","text":"Target/task: "+objective+". Images are ordered before the grip, after lift, and after a brief stationary closed-gripper hold."}]
    audit=[]
    for name,stage in stages:
        paths=stage.get("image_paths",[])
        if not isinstance(paths,list) or not paths or len(paths)>2:
            raise ValueError("Each verification stage needs one or two actual RGB images")
        for i,path in enumerate(paths):
            raw=Path(path).read_bytes()
            image=Image.open(io.BytesIO(raw)).convert("RGB")
            image.thumbnail((960,960))
            buffer=io.BytesIO(); image.save(buffer,format="PNG")
            label=name+" camera "+str(i+1)
            blocks.extend([{"type":"text","text":label},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")}}])
            audit.append({"stage":name,"camera_index":i,"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()})
    system=("You are an independent visual pickup verifier, not the robot controller. "
            "Decide from the ordered actual RGB observations whether the intended object was picked up off its support, "
            "is held by the gripper after the lift, and remains held after the brief hold. "
            "A closed hand, apparent contact, or object occlusion alone is insufficient. "
            "Say uncertain if the object identity, lift or retention cannot be seen. "
            "You receive no actor assessment, commanded result or hidden simulator state. "
            "Return only JSON with observed_pickup_hold (yes, no, or uncertain) and a short reason. "
            "Do not execute or recommend actions.")
    body={"model":config.model,"temperature":0,"max_tokens":2048,
          "messages":[{"role":"system","content":system},{"role":"user","content":blocks}],
          "tools":[{"type":"function","function":{"name":"report_pickup","description":"Report only the independent visual pickup assessment; no robot action.",
                    "parameters":{"type":"object","properties":{"observed_pickup_hold":{"type":"string","enum":["yes","no","uncertain"]},"reason":{"type":"string"}},
                                  "required":["observed_pickup_hold","reason"],"additionalProperties":False}}}],
          "tool_choice":"required","parallel_tool_calls":False}
    body.update(config.request_fields())
    request=urllib.request.Request(config.endpoint,data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+api_key,"Content-Type":"application/json"},method="POST")
    started=time.monotonic()
    with urllib.request.urlopen(request,timeout=180) as response:
        payload=json.loads(response.read())
    choice=payload["choices"][0]
    message=choice["message"]
    record={**config.metadata(),"input_images":audit,"fresh_request":True,"actor_assessment_sent":False,
            "elapsed_s":time.monotonic()-started,"usage":payload.get("usage"),"finish_reason":choice.get("finish_reason"),
            "returned_content":message.get("content"),"returned_tool_calls":message.get("tool_calls")}
    if audit_path is not None:
        Path(audit_path).write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    calls=message.get("tool_calls") or []
    if len(calls)!=1 or calls[0].get("function",{}).get("name")!="report_pickup":
        raise ValueError("Verifier did not return exactly one report_pickup function result")
    answer=json.loads(calls[0]["function"]["arguments"])
    record["response"]=answer
    return answer,record


def _gripper_measurement(evidence, name):
    stage = _stage(evidence, name)
    saved = stage.get("gripper_measurement")
    if isinstance(saved, dict):
        measurement = classify_gripper_fraction(saved.get("raw_value"),
            source=saved.get("source", "recorded_gripper_measurement"), stage=name)
        measurement["raw_type"] = saved.get("raw_type", measurement["raw_type"])
        # A failed getter has no raw scalar, but is invalid rather than missing.
        if saved.get("validity") == "invalid":
            measurement.update(validity="invalid", value=None, empty=None, nonempty=None)
        if saved.get("error"):
            measurement["error"] = saved["error"]
        return measurement
    fractions = evidence.get("gripper_fractions") or {}
    if isinstance(fractions, dict) and name in fractions:
        raw, source = fractions[name], "evidence.gripper_fractions"
    else:
        raw, source = stage.get("gripper_fraction"), "observation.gripper_fraction"
    return classify_gripper_fraction(raw, source=source, stage=name)


def _image_evidence(stages):
    """Require actual decodable stage images, including for injected test judges."""
    unavailable = []
    for name, stage in stages:
        paths = stage.get("image_paths")
        if not isinstance(paths, list) or not 1 <= len(paths) <= 2:
            unavailable.append({"stage": name, "status": "missing" if not paths else "invalid"})
            continue
        try:
            for path in paths:
                with Image.open(path) as image:
                    image.verify()
        except Exception as exc:
            unavailable.append({"stage": name, "status": "invalid", "error": type(exc).__name__})
    return unavailable


def verify_first_grasp(*, connector=None, baseline=None, evidence, output_dir, model=None,
                       api_key=None, objective="Pick up the intended object", visual_judge=None, provider_config=None):
    evidence=evidence or {}
    closed=_stage(evidence,"post_close"); lifted=_stage(evidence,"post_lift"); held=_stage(evidence,"post_hold")
    lift=evidence.get("measured_lift_m")
    lift_source="explicit_actual_measurement"
    if not _finite(lift):
        a,b=_z(closed),_z(lifted)
        # Unbound frame fallback may select a later object's lift. A
        # per-execution record must remain unknown if its own samples are absent.
        lift=b-a if a is not None and b is not None else (
            None if evidence.get('execution_ref') else _recorded_lift(output_dir))
        lift_source="post_close_to_post_lift" if a is not None and b is not None else "recorded_lift_segment_samples"
    duration=evidence.get("measured_hold_s")
    if not _finite(duration):
        a,b=_clock(lifted),_clock(held)
        duration=b-a if a is not None and b is not None else None
    measurements=[_gripper_measurement(evidence,name) for name in ("post_lift","post_hold")]
    opens=[measurement["value"] for measurement in measurements]
    command_status=evidence.get("execution_status",evidence.get("command_status"))
    checks={"command_completed":command_status=="succeeded",
            "actor_reports_held":evidence.get("actor_visual_assessment")=="held",
            "measured_lift":_finite(lift) and lift>=.04,
            "measured_hold":_finite(duration) and duration>=.9-1e-6,
            "nonempty_gripper":all(m["nonempty"] is True for m in measurements)}
    unknown_checks=[]
    if command_status not in ("succeeded","failed"):
        unknown_checks.append("command_completed")
    if not _finite(lift):
        unknown_checks.append("measured_lift")
    if not _finite(duration):
        unknown_checks.append("measured_hold")
    if any(m["validity"] != "valid" for m in measurements):
        unknown_checks.append("nonempty_gripper")
    result={"evaluated":False,"first_grasp_success":False,"outcome":"unknown",
            "checks":checks,"unknown_checks":unknown_checks,"measured_lift_m":lift,
            "lift_measurement_source":lift_source,"measured_hold_s":duration,"gripper_fractions":opens,
            "gripper_measurements":measurements,"visual_evaluated":False,
            "actor_visual_assessment":evidence.get("actor_visual_assessment"),"visual_assessment":None,"error":None,
            "scope":"Sensor/proprioception plus fresh independent visual judge; not simulation ground truth or full task success."}
    stages=[(name,_stage(evidence,name)) for name in ("pre_grasp","post_lift","post_hold")]
    unavailable=_image_evidence(stages)
    if unavailable:
        result.update(visual_evidence_errors=unavailable,
                      reason="Missing or invalid stage images; independent visual pickup assessment not evaluated")
        return result
    try:
        if visual_judge is not None:
            answer=visual_judge(stages,objective=objective)
            audit={"test_or_injected_judge":True,"response":answer}
        else:
            if not api_key or not model:
                raise ValueError("Independent visual verifier requires explicit provider credentials/model")
            answer,audit=_visual_request(stages,model=model,api_key=api_key,objective=objective,audit_path=Path(output_dir)/"grasp_visual_response.json",provider_config=provider_config)
        if not isinstance(answer,dict) or answer.get("observed_pickup_hold") not in ("yes","no","uncertain"):
            raise ValueError("Visual verifier returned invalid assessment")
        result["visual_assessment"]={"observed_pickup_hold":answer["observed_pickup_hold"],"reason":str(answer.get("reason",""))[:1024]}
        path=Path(output_dir)/"grasp_visual_verifier.json"
        path.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
        result["visual_evaluated"]=True
        result["evaluated"]=not unknown_checks
        # Actor narration is diagnostic only and never gates the independent judge.
        telemetry_passed=all(checks[key] for key in ("command_completed","measured_lift","measured_hold","nonempty_gripper"))
        result["first_grasp_success"]=result["evaluated"] and telemetry_passed and answer["observed_pickup_hold"]=="yes"
        if result["first_grasp_success"]:
            result.update(outcome="success",reason="Measured pickup/hold and independent visual evidence agree")
        elif unknown_checks:
            result["reason"]="Missing or invalid pickup/hold telemetry; no success inferred"
        elif not telemetry_passed or answer["observed_pickup_hold"]=="no":
            result.update(outcome="failure",reason="Pickup/hold evidence does not meet verification requirements")
        else:
            result["reason"]="Independent visual pickup/hold assessment is uncertain"
    except Exception as exc:
        result.update(evaluated=False,error=type(exc).__name__,first_grasp_success=False,
                      reason="Independent visual pickup assessment could not be completed")
    return result
