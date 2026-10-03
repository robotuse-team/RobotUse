"""Local web interface for selecting, running, and inspecting episodes."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os

import gradio as gr

from src.ui.runner import EpisodeRunner, display_text
from src.agent.playbook import DEFAULT_VERSION as DEFAULT_PLAYBOOK_VERSION, VERSION_CHOICES
from src.utils.logging_utils import configure_logging, redact_data

CSS = """
.gradio-container { width: 100% !important; max-width: 1320px !important; min-width: 0 !important; margin: auto; }
#masthead { border-bottom: 1px solid #d8dcd8; padding: 10px 0 24px; margin-bottom: 20px; }
#masthead .eyebrow { font-size: 12px; letter-spacing: .16em; color: #526259; font-weight: 600; }
#masthead h1 { font-size: 30px; font-weight: 550; letter-spacing: -.04em; margin: 8px 0; }
#masthead p { color: #626b65; font-size: 14px; margin: 0; }
#run-button { background: #233c30; border-color: #233c30; color: white; }
#task-instruction { min-height: 74px; color: #525e56; padding: 8px 2px; }
#process-status textarea, #verifier-status textarea { font-weight: 550; }
footer { display: none !important; }
"""


def build_app(runner: EpisodeRunner | None = None) -> gr.Blocks:
    runner = runner or EpisodeRunner()
    tasks = sorted(runner.catalog)
    default = "BananaInBowlTask" if "BananaInBowlTask" in tasks else tasks[0]

    def description(task):
        row = runner.catalog[task]
        return f"{row['instruction']}\n\n{row['difficulty_label'].capitalize()} · {row['episode_s']} s simulation budget"

    def refresh(episode_id):
        state = runner.snapshot(episode_id)
        details = redact_data(state["details"])
        videos = [gr.update(value=path, visible=bool(path)) for path in state["videos"]]
        previews = [gr.update(value=frame, visible=not video)
                    for frame, video in zip(state["previews"], state["videos"])]
        return (display_text(state["status"]), state["verifier"], display_text(state["logs"]),
                details, *videos, display_text(state["activity"]), *previews)

    def start(task, seed, playbook, mode):
        try:
            selected = runner.start(task, seed, playbook, mode == "Configuration check")
        except (ValueError, RuntimeError, TypeError) as exc:
            raise gr.Error(display_text(str(exc))) from exc
        return gr.update(choices=runner.episodes(), value=selected)

    def refresh_history(episode_id, turn_id, follow_latest, shown):
        view = runner.history(episode_id, turn_id, follow_latest=follow_latest)
        detail_keys = ("selected", "input", "output", "results", "input_images", "result_images", "full_input")
        signature = dict(
            detail=hashlib.sha256(json.dumps([episode_id, {k: view[k] for k in detail_keys}], sort_keys=True).encode()).hexdigest(),
            navigation=hashlib.sha256(json.dumps([view["choices"], view["selected"], view["summary"], follow_latest]).encode()).hexdigest())
        if signature == shown:
            return (gr.skip(),) * 12  # Preserve expanded JSON and image previews while browsing.
        result = [gr.update(choices=view["choices"], value=view["selected"], interactive=bool(view["count"])),
                view["summary"], view["input"], view["output"], view["results"],
                gr.update(value=view["input_images"], visible=bool(view["input_images"])),
                gr.update(value=view["result_images"], visible=bool(view["result_images"])),
                view["full_input"], gr.update(interactive=view["index"] > 0),
                gr.update(interactive=view["index"] < view["count"] - 1),
                gr.update(value=follow_latest), signature]
        if isinstance(shown, dict) and shown.get("detail") == signature["detail"]:
            result[2:8] = [gr.skip()] * 6
        return result

    def load_history(episode_id):
        return refresh_history(episode_id, None, True, "")

    def select_turn(episode_id, turn_id):
        return refresh_history(episode_id, turn_id, False, "")

    def move_turn(episode_id, turn_id, offset):
        view = runner.history(episode_id, turn_id, offset=offset)
        return select_turn(episode_id, view["selected"])

    with gr.Blocks(title="RobotUse · Episodes", analytics_enabled=False) as app:
        gr.HTML('<header id="masthead"><div class="eyebrow">ROBOTUSE</div>'
                '<h1>Episode runner</h1><p>Choose a task, run it in RoboLab, and inspect the native verifier.</p></header>')
        with gr.Row():
            with gr.Column(scale=4, min_width=300):
                task = gr.Dropdown(tasks, value=default, label="Task", filterable=True)
                instruction = gr.Markdown(description(default), elem_id="task-instruction")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, minimum=0, maximum=2**32-1, label="Seed")
                    playbook = gr.Dropdown(list(VERSION_CHOICES), value=f"v{DEFAULT_PLAYBOOK_VERSION}", label="Playbook")
                mode = gr.Radio(["Run episode", "Configuration check"], value="Configuration check", label="Mode")
                gr.Markdown("Episode runs use the provider and model configured in your environment.", elem_classes="hint")
                with gr.Row():
                    run_button = gr.Button("Start", variant="primary", elem_id="run-button")
                    stop_button = gr.Button("Stop")
                selected = gr.Dropdown(runner.episodes(), label="Recent episodes", value=None)
                reload_button = gr.Button("Refresh list", size="sm")
            with gr.Column(scale=7, min_width=280):
                with gr.Row():
                    status = gr.Textbox(value="Ready", label="Process", lines=2, max_lines=4, interactive=False, elem_id="process-status")
                    verdict = gr.Textbox(value="Not evaluated", label="Native verifier", lines=2, interactive=False, elem_id="verifier-status")
                activity = gr.Textbox(label="Activity", interactive=False, lines=1)
                with gr.Tabs():
                    with gr.Tab("Front camera"):
                        front_live = gr.Image(label="Live front camera", format="jpeg", interactive=False, height=340)
                        front = gr.Video(label="Front camera", interactive=False, height=340, visible=False)
                    with gr.Tab("Wrist camera"):
                        wrist_live = gr.Image(label="Live wrist camera", format="jpeg", interactive=False, height=340)
                        wrist = gr.Video(label="Wrist camera", interactive=False, height=340, visible=False)
                    with gr.Tab("History"):
                        with gr.Row():
                            previous_turn = gr.Button("Previous", size="sm", scale=0, min_width=80, interactive=False)
                            turn = gr.Dropdown([], label="Turn", interactive=False)
                            next_turn = gr.Button("Next", size="sm", scale=0, min_width=80, interactive=False)
                        follow_latest = gr.Checkbox(value=True, label="Follow latest")
                        history_summary = gr.Markdown("No recorded turns yet.", elem_id="history-summary")
                        input_images = gr.Gallery(label="Input images", columns=2, rows=1, height=260,
                            object_fit="contain", interactive=False, visible=False, elem_id="history-input-images")
                        with gr.Row():
                            turn_input = gr.JSON(label="Input", max_height=300, elem_id="history-input")
                            turn_output = gr.JSON(label="LLM output", max_height=300, elem_id="history-output")
                        tool_results = gr.JSON(label="Tool result", max_height=300, elem_id="history-tool-result")
                        result_images = gr.Gallery(label="Tool result images", columns=2, rows=1, height=260,
                            object_fit="contain", interactive=False, visible=False, elem_id="history-result-images")
                        with gr.Accordion("Recorded LLM input", open=False):
                            full_input = gr.JSON(label="Recorded request", max_height=500)
                        history_shown = gr.State({})
                    with gr.Tab("Result"):
                        details = gr.JSON(label="Episode and verifier")
                gr.Markdown("Live views refresh every second and stay still while the simulator waits. "
                            "Full recordings appear after completion. Task success comes from the native verifier.")
        with gr.Accordion("Process log", open=False):
            logs = gr.Textbox(label="Recent output", lines=12, max_lines=18, interactive=False)
        outputs = [status, verdict, logs, details, front, wrist, activity, front_live, wrist_live]
        task.change(description, task, instruction, queue=False, api_visibility="private")
        run_button.click(start, [task, seed, playbook, mode], selected, queue=False, api_name="start_episode")
        stop_button.click(runner.stop, selected, queue=False, api_name="stop_episode")
        reload_button.click(lambda: gr.update(choices=runner.episodes()), outputs=selected, queue=False, api_visibility="private")
        selected.change(refresh, selected, outputs, queue=False, api_name="episode_status")
        timer = gr.Timer(1)
        timer.tick(refresh, selected, outputs, queue=False, api_visibility="private")
        history_outputs = [turn, history_summary, turn_input, turn_output, tool_results,
                           input_images, result_images, full_input, previous_turn, next_turn,
                           follow_latest, history_shown]
        selected.change(load_history, selected, history_outputs, queue=False, api_name="episode_history")
        turn.input(select_turn, [selected, turn], history_outputs, queue=False, api_name="history_turn")
        previous_turn.click(lambda episode, turn_id: move_turn(episode, turn_id, -1),
            [selected, turn], history_outputs, queue=False, api_name="previous_turn")
        next_turn.click(lambda episode, turn_id: move_turn(episode, turn_id, 1),
            [selected, turn], history_outputs, queue=False, api_name="next_turn")
        follow_latest.input(refresh_history, [selected, turn, follow_latest, history_shown],
            history_outputs, queue=False, api_visibility="private")
        timer.tick(refresh_history, [selected, turn, follow_latest, history_shown],
            history_outputs, queue=False, api_visibility="private")
    atexit.register(runner.close)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    configure_logging()
    theme = gr.themes.Base(primary_hue="green", neutral_hue="gray", font=["Arial", "sans-serif"],
                           font_mono=["monospace"], radius_size="sm").set(
        body_background_fill="#f8f9f6", block_background_fill="#ffffff",
        block_border_color="#dce1dc", button_primary_background_fill="#233c30")
    runner = EpisodeRunner()
    build_app(runner).queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port,
        share=False, inbrowser=False, footer_links=[], theme=theme, css=CSS, show_error=False,
        allowed_paths=[str(runner.runs_root)], ssr_mode=False)


if __name__ == "__main__":
    main()
