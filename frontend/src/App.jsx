import { useState } from "react";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function SourceList({ hits }) {
  if (!hits?.length) return null;
  return (
    <div className="sources">
      <h3>Sources</h3>
      <ul>
        {hits.map((h, i) => (
          <li key={i}>
            <code>[{h.doc_id}:{h.page}]</code> — <span title={h.source_path}>{h.chunk_id}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [k, setK] = useState(6);
  const [rerank, setRerank] = useState(true);
  const [temp, setTemp] = useState(0.2);
  const [error, setError] = useState("");

  async function ask(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    setAnswer("");
    setHits([]);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          k: Number(k),
          rerank,
          top_m: 40,
          max_tokens: 512,
          temperature: Number(temp),
        }),
      });
      const data = await res.json();

      if (data.status === "blocked") {
        setError(`Blocked: ${data.reason}`);
      } else if (data.status === "no_context") {
        setAnswer(data.answer || "I don't have enough information in the provided documents.");
        setHits(data.hits || []);
      } else if (data.status === "ok") {
        setAnswer(data.answer || "");
        setHits(data.hits || []);
      } else {
        setError("Unexpected response from server.");
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", fontFamily: "Inter, system-ui, sans-serif" }}>
      <h1>🧠 DocuChat Pro</h1>
      <p style={{ color: "#555" }}>Ask your documents — grounded answers with citations.</p>

      <form onSubmit={ask} style={{ display: "grid", gap: 12, marginBottom: 18 }}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder='Ask e.g. "What is retrieval-augmented generation?"'
          rows={3}
          style={{ padding: 12, borderRadius: 8, border: "1px solid #ddd" }}
        />
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <label>k:
            <input type="number" min={1} max={12} value={k}
              onChange={(e) => setK(e.target.value)}
              style={{ marginLeft: 6, width: 70 }} />
          </label>
          <label>temperature:
            <input type="number" step="0.1" min={0} max={1} value={temp}
              onChange={(e) => setTemp(e.target.value)}
              style={{ marginLeft: 6, width: 90 }} />
          </label>
          <label>
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            &nbsp;rerank
          </label>
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "8px 16px",
              borderRadius: 8,
              border: "1px solid #222",
              background: "#111",
              color: "#fff",
              cursor: loading ? "wait" : "pointer"
            }}
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>
      </form>

      {error && <div style={{ color: "#b00020", marginBottom: 12 }}>⚠️ {error}</div>}

      {answer && (
        <div style={{ background: "#fafafa", border: "1px solid #eee", borderRadius: 8, padding: 16 }}>
          <h3>Answer</h3>
          <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{answer}</div>
        </div>
      )}

      <SourceList hits={hits} />

      <footer style={{ marginTop: 24, color: "#888", fontSize: 12 }}>
        Backend: {API} • Toggle rerank or tune k/temp to compare results.
      </footer>
    </div>
  );
}
