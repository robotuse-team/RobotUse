import { useState } from "react";

interface Props {
  data: unknown;
  depth?: number;
}

export function JsonTree({ data, depth = 0 }: Props) {
  if (data === null || data === undefined) {
    return <span style={{ color: "#95a5a6" }}>null</span>;
  }

  if (typeof data === "boolean") {
    return <span style={{ color: "#e67e22" }}>{String(data)}</span>;
  }

  if (typeof data === "number") {
    return <span style={{ color: "#3498db" }}>{data}</span>;
  }

  if (typeof data === "string") {
    if (data.length > 200) {
      return <StringValue value={data} />;
    }
    return <span style={{ color: "#2ecc71" }}>"{data}"</span>;
  }

  if (Array.isArray(data)) {
    return <ArrayView items={data} depth={depth} />;
  }

  if (typeof data === "object") {
    return <ObjectView obj={data as Record<string, unknown>} depth={depth} />;
  }

  return <span>{String(data)}</span>;
}

function StringValue({ value }: { value: string }) {
  const [expanded, setExpanded] = useState(false);
  const display = expanded ? value : value.slice(0, 100) + "...";
  return (
    <span
      style={{ color: "#2ecc71", cursor: "pointer" }}
      onClick={() => setExpanded(!expanded)}
    >
      "{display}"
    </span>
  );
}

function ArrayView({ items, depth }: { items: unknown[]; depth: number }) {
  const [collapsed, setCollapsed] = useState(depth > 1 && items.length > 3);

  if (items.length === 0) return <span style={{ color: "#95a5a6" }}>[]</span>;

  return (
    <div style={{ marginLeft: depth > 0 ? 12 : 0 }}>
      <span
        style={{ color: "#95a5a6", cursor: "pointer" }}
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? "▶" : "▼"} [{items.length}]
      </span>
      {!collapsed &&
        items.map((item, i) => (
          <div key={i} style={{ marginLeft: 12 }}>
            <span style={{ color: "#95a5a6", fontSize: 10 }}>{i}: </span>
            <JsonTree data={item} depth={depth + 1} />
          </div>
        ))}
    </div>
  );
}

function ObjectView({
  obj,
  depth,
}: {
  obj: Record<string, unknown>;
  depth: number;
}) {
  const keys = Object.keys(obj);
  const [collapsed, setCollapsed] = useState(depth > 1 && keys.length > 5);

  if (keys.length === 0) return <span style={{ color: "#95a5a6" }}>{"{}"}</span>;

  // "__proto__" is reserved in JS — use bracket access with a different key name
  const protoTag = typeof obj["__proto_type__"] === "string" ? obj["__proto_type__"] : undefined;

  return (
    <div style={{ marginLeft: depth > 0 ? 12 : 0 }}>
      <span
        style={{ color: "#95a5a6", cursor: "pointer" }}
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? "▶" : "▼"}{" "}
        {protoTag ? (
          <span style={{ color: "#ff9f6e", fontSize: 10 }}>
            {protoTag.split(".").pop()}
          </span>
        ) : (
          `{${keys.length}}`
        )}
      </span>
      {!collapsed &&
        keys
          .filter((k) => k !== "__proto_type__")
          .map((key) => (
            <div key={key} style={{ marginLeft: 12 }}>
              <span style={{ color: "#d7e2f2" }}>{key}: </span>
              <JsonTree data={obj[key]} depth={depth + 1} />
            </div>
          ))}
    </div>
  );
}
