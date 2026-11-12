import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  GitBranch,
  Layers3,
  LockKeyhole,
  Package,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  X,
} from "lucide-react";
import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Feedback = {
  id: string;
  input_text: string;
  original_prediction?: Record<string, unknown> | null;
  corrected_output: Record<string, unknown>;
  status: string;
  source: string;
  privacy_flags: string[];
  conflict: number;
};
type Dataset = {
  id: string;
  manifest: {
    counts: Record<string, number>;
    warnings: string[];
    file_sha256: Record<string, string>;
    schema_version: string;
    statistics?: Record<string, unknown>;
    contamination?: Record<string, unknown>;
  };
  created_at: string;
};
type Model = {
  id: string;
  base_model: string;
  status: string;
  metrics: Record<string, unknown>;
  dataset_id?: string;
  evaluation_id?: string;
};
type Experiment = {
  id: string;
  status: string;
  dataset_id: string;
  config: Record<string, unknown>;
  manifest: Record<string, unknown>;
};
type EvalRun = { id: string; report: Record<string, any> };
type EvaluationSet = {
  id: string;
  manifest: { count: number; created_at: string };
};
type Deployment = {
  id: string;
  model_id: string;
  action: string;
  operator: string | null;
  reason: string | null;
  created_at: string;
  details: Record<string, any>;
};
const API = "/api/v1";
function authHeaders(): Record<string, string> {
  const key = sessionStorage.getItem(
    "llm-feedback-finetuning-pipeline-api-key",
  );
  return key ? { "X-LLM-Feedback-Key": key } : {};
}
async function get<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: authHeaders() });
  if (!response.ok)
    throw new Error((await response.text()) || response.statusText);
  return response.json();
}
async function upload<T>(url: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(url, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      data.detail?.message || data.detail?.error || JSON.stringify(data.detail),
    );
  return data;
}
async function post<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      data.detail?.message || data.detail?.error || JSON.stringify(data.detail),
    );
  return data;
}
export function App() {
  const [page, setPage] = useState("Overview"),
    [apiKey, setApiKey] = useState(
      () =>
        sessionStorage.getItem("llm-feedback-finetuning-pipeline-api-key") ||
        "",
    ),
    [feedback, setFeedback] = useState<Feedback[]>([]),
    [datasets, setDatasets] = useState<Dataset[]>([]),
    [evaluationSets, setEvaluationSets] = useState<EvaluationSet[]>([]),
    [models, setModels] = useState<Model[]>([]),
    [evaluations, setEvaluations] = useState<EvalRun[]>([]),
    [experiments, setExperiments] = useState<Experiment[]>([]),
    [deploymentHistory, setDeploymentHistory] = useState<Deployment[]>([]),
    [active, setActive] = useState<Model | null>(null),
    [query, setQuery] = useState(""),
    [prediction, setPrediction] = useState<Record<string, unknown> | null>(
      null,
    ),
    [text, setText] = useState(
      "Men’s waterproof hiking jacket, navy blue, size XL, polyester shell.",
    ),
    [busy, setBusy] = useState(false),
    [connected, setConnected] = useState(false),
    [correction, setCorrection] = useState(""),
    [selectedDataset, setSelectedDataset] = useState(""),
    [notice, setNotice] = useState("");
  const refresh = async () => {
    try {
      const [f, d, q, m, a, e, x, h] = await Promise.all([
        get<Feedback[]>(`${API}/feedback`),
        get<Dataset[]>(`${API}/datasets`),
        get<EvaluationSet[]>(`${API}/evaluation-sets`),
        get<Model[]>(`${API}/models`),
        get<Model | null>(`${API}/models/active`),
        get<EvalRun[]>(`${API}/evaluations`),
        get<Experiment[]>(`${API}/experiments`),
        get<Deployment[]>(`${API}/deployments`),
      ]);
      setFeedback(f);
      setDatasets(d);
      setEvaluationSets(q);
      setModels(m);
      setActive(a);
      setEvaluations(e);
      setExperiments(x);
      setDeploymentHistory(h);
      if (!selectedDataset && d.length) setSelectedDataset(d[0].id);
      setConnected(true);
      setNotice("");
    } catch (e) {
      setConnected(false);
      setNotice(String(e));
    }
  };
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 6000);
    return () => clearInterval(id);
  }, []);
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      await refresh();
      setNotice("Changes saved.");
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy(false);
    }
  };
  const nav = [
    ["Overview", Activity],
    ["Corrections", GitBranch],
    ["Datasets", Database],
    ["Experiments", FlaskConical],
    ["Evaluation", Layers3],
    ["Registry", Boxes],
    ["Playground", Sparkles],
  ] as const;
  const pending = feedback.filter((x) => x.status === "pending").length,
    latestEval = evaluations[0]?.report;
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark">
            <Package size={20} />
          </div>
          <div>
            <b>LLM Feedback Fine-Tuning Pipeline</b>
            <small>MODEL OPERATIONS</small>
          </div>
        </div>
        <div className="workspace">
          <span className="pulse" />
          LOCAL WORKSPACE
          <ChevronRight size={14} />
        </div>
        <nav>
          {nav.map(([label, Icon]) => (
            <button
              key={label}
              className={page === label ? "selected" : ""}
              onClick={() => setPage(label)}
            >
              <Icon size={17} />
              <span>{label}</span>
              {label === "Corrections" && pending > 0 && <em>{pending}</em>}
            </button>
          ))}
        </nav>
        <div className="side-bottom">
          <div className="privacy">
            <LockKeyhole size={15} />
            <span>
              <b>Data stays local</b>
              <small>No telemetry · no uploads</small>
            </span>
          </div>
          <div className="side-foot">
            LLM FEEDBACK PIPELINE <span>v0.1.0</span>
          </div>
        </div>
      </aside>
      <main>
        <header>
          <div className="crumb">
            WORKSPACE <span>/</span> {page.toUpperCase()}
          </div>
          <div className="header-right">
            <input
              className="api-key-input"
              aria-label="Optional API key"
              type="password"
              placeholder="API key (optional)"
              value={apiKey}
              onChange={(event) => {
                const value = event.target.value;
                setApiKey(value);
                if (value)
                  sessionStorage.setItem(
                    "llm-feedback-finetuning-pipeline-api-key",
                    value,
                  );
                else
                  sessionStorage.removeItem(
                    "llm-feedback-finetuning-pipeline-api-key",
                  );
              }}
              onBlur={() => void refresh()}
            />
            <span className="local">
              <i />
              LOCAL MODE
            </span>
            <button className="icon-btn" title="Refresh" onClick={refresh}>
              <RefreshCw size={16} />
            </button>
            <div className="avatar">OP</div>
          </div>
        </header>
        <div className="content">
          <div className="title-row">
            <div>
              <div className="eyebrow">CORRECTION TO DEPLOYMENT</div>
              <h1>{page}</h1>
              <p>
                {page === "Overview"
                  ? "A measurable learning loop for product catalog quality."
                  : page === "Corrections"
                    ? "Review and govern every correction before it enters training."
                    : page === "Datasets"
                      ? "Immutable, traceable training snapshots."
                      : page === "Playground"
                        ? "Try the active local model and submit a correction."
                        : "Operational evidence for local model adaptation."}
              </p>
            </div>
            <button className="quiet" onClick={refresh}>
              <RefreshCw size={15} /> Refresh
            </button>
          </div>
          {notice && (
            <div
              className={
                notice === "Changes saved." ? "notice success" : "notice"
              }
            >
              {notice}
            </div>
          )}
          {page === "Overview" && (
            <>
              <div className="metrics">
                <Metric
                  label="PENDING REVIEW"
                  value={pending.toString().padStart(2, "0")}
                  icon={<GitBranch />}
                  detail="Corrections awaiting approval"
                />
                <Metric
                  label="DATASET VERSIONS"
                  value={String(datasets.length).padStart(2, "0")}
                  icon={<Database />}
                  detail="Immutable snapshots built"
                />
                <Metric
                  label="REGISTERED MODELS"
                  value={String(models.length).padStart(2, "0")}
                  icon={<Boxes />}
                  detail="Candidate and deployed"
                />
                <Metric
                  label="ACTIVE MODEL"
                  value={active ? "READY" : "—"}
                  icon={<ShieldCheck />}
                  detail={active?.id || "No production model selected"}
                />
              </div>
              <section className="panel pipeline">
                <div className="panel-head">
                  <div>
                    <h2>Adaptation pipeline</h2>
                    <p>Current state across the correction loop</p>
                  </div>
                  <span className="tag neutral">LIVE STATUS</span>
                </div>
                <div className="steps">
                  {[
                    [
                      "Feedback",
                      String(feedback.length),
                      "Corrections captured",
                    ],
                    ["Review", String(pending) + " pending", "Human approval"],
                    [
                      "Dataset",
                      String(datasets.length) + " versions",
                      "Reproducible split",
                    ],
                    ["Train", String(models.length) + " models", "Local LoRA"],
                    ["Evaluate", "—", "Evidence required"],
                    [
                      "Deploy",
                      active ? "Active" : "Not deployed",
                      active?.id || "Promotion gated",
                    ],
                  ].map(([a, b, c], i) => (
                    <div className="step" key={a}>
                      <div
                        className={
                          "step-icon " +
                          (i === 0 && feedback.length
                            ? "complete"
                            : i === 1 && pending
                              ? "attention"
                              : "")
                        }
                      >
                        <span>
                          {i === 0 && feedback.length ? (
                            <Check size={16} />
                          ) : (
                            i + 1
                          )}
                        </span>
                      </div>
                      <div>
                        <b>{a}</b>
                        <strong>{b}</strong>
                        <small>{c}</small>
                      </div>
                      {i < 5 && (
                        <ChevronRight className="step-arrow" size={15} />
                      )}
                    </div>
                  ))}
                </div>
              </section>
              <div className="lower-grid">
                <section className="panel">
                  <div className="panel-head">
                    <div>
                      <h2>Review queue</h2>
                      <p>Most recent corrections</p>
                    </div>
                    <button
                      className="link"
                      onClick={() => setPage("Corrections")}
                    >
                      View all <ChevronRight size={14} />
                    </button>
                  </div>
                  <div className="mini-list">
                    {feedback.slice(0, 4).map((f) => (
                      <div className="mini-row" key={f.id}>
                        <span className="dot amber" />
                        <div>
                          <b>{f.input_text}</b>
                          <small>
                            {f.source} · {f.status}
                          </small>
                        </div>
                        <ChevronRight size={15} />
                      </div>
                    ))}
                    {feedback.length === 0 && (
                      <Empty
                        icon={<CircleHelp />}
                        title="Nothing in review yet"
                        detail="Load the synthetic fixture or import correction records."
                      />
                    )}
                  </div>
                </section>
                <section className="panel">
                  <div className="panel-head">
                    <div>
                      <h2>Model readiness</h2>
                      <p>Promotion requires measured evaluation</p>
                    </div>
                    <span className={"tag " + (latestEval ? "good" : "warn")}>
                      {latestEval ? "EVALUATED" : "NOT EVALUATED"}
                    </span>
                  </div>
                  <div className="readiness">
                    <div className="readiness-score">
                      {latestEval
                        ? Math.round(
                            Number(latestEval.candidate.exact_match_accuracy) *
                              100,
                          ) + "%"
                        : "—"}
                      <small>{latestEval ? "EXACT MATCH" : "NO RESULT"}</small>
                    </div>
                    <div className="readiness-copy">
                      <b>
                        {latestEval
                          ? "Paired evaluation recorded"
                          : "No evaluation attached"}
                      </b>
                      <p>
                        {latestEval
                          ? `${latestEval.improved} improved · ${latestEval.regressed} regressed · n=${latestEval.candidate.n}`
                          : "LLM Feedback Fine-Tuning Pipeline blocks promotion until evaluation evidence and valid artifacts exist."}
                      </p>
                      <button
                        className="text-btn"
                        onClick={() => setPage("Evaluation")}
                      >
                        Open evaluation <ChevronRight size={14} />
                      </button>
                    </div>
                  </div>
                </section>
              </div>
              <section className="panel bottom-note">
                <LockKeyhole size={17} />
                <div>
                  <b>Local-first by default</b>
                  <p>
                    Demo outputs are labeled simulated. No benchmark or model
                    improvement is reported without real measurements.
                  </p>
                </div>
              </section>
            </>
          )}
          {page === "Corrections" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2>Correction queue</h2>
                  <p>
                    {feedback.length} records · imported records require
                    explicit review
                  </p>
                </div>
                <input
                  className="search"
                  placeholder="Filter product text…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>PRODUCT INPUT</th>
                      <th>ORIGINAL PREDICTION</th>
                      <th>HUMAN CORRECTION</th>
                      <th>SOURCE / STATUS</th>
                      <th>FLAGS</th>
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {feedback
                      .filter((f) =>
                        f.input_text
                          .toLowerCase()
                          .includes(query.toLowerCase()),
                      )
                      .map((f) => (
                        <tr key={f.id}>
                          <td className="input-cell">
                            {f.input_text}
                            <small>{f.id}</small>
                          </td>
                          <td>
                            <code>
                              {JSON.stringify(f.original_prediction) || "—"}
                            </code>
                          </td>
                          <td>
                            <code>{JSON.stringify(f.corrected_output)}</code>
                          </td>
                          <td>
                            <span
                              className={
                                "tag " +
                                (f.status === "approved"
                                  ? "good"
                                  : f.status === "pending"
                                    ? "warn"
                                    : "neutral")
                              }
                            >
                              {f.status}
                            </span>
                            <small>{f.source}</small>
                          </td>
                          <td>
                            {f.privacy_flags?.length > 0 ? (
                              <>
                                <span className="tag danger">
                                  {f.privacy_flags.join(", ")}
                                </span>
                                <button
                                  className="quiet"
                                  onClick={() =>
                                    act(() =>
                                      post(`${API}/feedback/${f.id}/redact`, {
                                        reviewer: "dashboard-operator",
                                      }),
                                    )
                                  }
                                >
                                  Redact
                                </button>
                              </>
                            ) : f.conflict ? (
                              <span className="tag danger">CONFLICT</span>
                            ) : (
                              <span className="muted">—</span>
                            )}
                          </td>
                          <td>
                            {f.status === "quarantined" && f.conflict ? (
                              <button
                                className="quiet"
                                disabled={busy}
                                onClick={() => {
                                  const reason = window.prompt(
                                    "Explain why this correction is the authoritative choice.",
                                  );
                                  if (reason)
                                    void act(() =>
                                      post(
                                        `${API}/feedback/${f.id}/resolve-conflict`,
                                        {
                                          reviewer: "dashboard-operator",
                                          reason,
                                        },
                                      ),
                                    );
                                }}
                              >
                                Resolve conflict
                              </button>
                            ) : f.status === "pending" ? (
                              <div className="actions">
                                <button
                                  className="approve"
                                  disabled={
                                    busy ||
                                    !!f.privacy_flags?.length ||
                                    !!f.conflict
                                  }
                                  title={
                                    f.privacy_flags?.length
                                      ? "Review/redact privacy flags first"
                                      : ""
                                  }
                                  onClick={() =>
                                    act(() =>
                                      post(`${API}/feedback/${f.id}/review`, {
                                        decision: "approved",
                                        reviewer: "dashboard-operator",
                                      }),
                                    )
                                  }
                                >
                                  <Check size={14} /> Approve
                                </button>
                                <button
                                  className="reject"
                                  disabled={busy}
                                  onClick={() =>
                                    act(() =>
                                      post(`${API}/feedback/${f.id}/review`, {
                                        decision: "rejected",
                                        reviewer: "dashboard-operator",
                                      }),
                                    )
                                  }
                                >
                                  <X size={14} />
                                </button>
                              </div>
                            ) : (
                              <span className="muted">Reviewed</span>
                            )}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
                {feedback.length === 0 && (
                  <Empty
                    icon={<GitBranch />}
                    title="No feedback records"
                    detail="Use the CLI or API to import a CSV or JSONL correction file."
                  />
                )}
              </div>
            </section>
          )}
          {page === "Datasets" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2>Dataset versions</h2>
                  <p>
                    Built only from approved, unique, conflict-free feedback
                  </p>
                </div>
                <div className="actions">
                  <label className="quiet file-action">
                    Freeze independent holdout
                    <input
                      type="file"
                      accept=".csv,.jsonl,.ndjson"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file)
                          void act(() =>
                            upload(`${API}/evaluation-sets`, file),
                          );
                      }}
                    />
                  </label>
                  <button
                    className="primary"
                    disabled={busy}
                    onClick={() => act(() => post(`${API}/datasets/build`, {}))}
                  >
                    <Database size={15} /> Create dataset
                  </button>
                </div>
              </div>
              {datasets.length ? (
                <div className="dataset-list">
                  {datasets.map((d) => (
                    <article className="dataset" key={d.id}>
                      <div className="dataset-icon">
                        <Database size={18} />
                      </div>
                      <div className="dataset-main">
                        <b>{d.id}</b>
                        <small>
                          Product extraction · schema{" "}
                          {d.manifest.schema_version} ·{" "}
                          {new Date(d.created_at).toLocaleString()}
                        </small>
                        <div className="split-bars">
                          {Object.entries(d.manifest.counts || {}).map(
                            ([name, count]) => (
                              <span key={name}>
                                {name}
                                <b>{count}</b>
                              </span>
                            ),
                          )}
                        </div>
                        {d.manifest.statistics && (
                          <small>
                            {String(
                              d.manifest.statistics.approved_examples || 0,
                            )}{" "}
                            approved · average input{" "}
                            {String(
                              d.manifest.statistics.average_input_characters ||
                                0,
                            )}{" "}
                            chars ·{" "}
                            {
                              Object.keys(
                                d.manifest.statistics.category_coverage || {},
                              ).length
                            }{" "}
                            categories
                          </small>
                        )}
                        {d.manifest.warnings?.map((warning) => (
                          <small className="tag warn" key={warning}>
                            {warning}
                          </small>
                        ))}
                      </div>
                      <span className="tag good">HASH VERIFIED ON USE</span>
                    </article>
                  ))}
                </div>
              ) : (
                <Empty
                  icon={<Database />}
                  title="No dataset versions yet"
                  detail="Approve at least three unique corrections, then build a version."
                />
              )}
            </section>
          )}
          {page === "Playground" && (
            <div className="play-grid">
              <section className="panel playground">
                <div className="panel-head">
                  <div>
                    <h2>Product extraction</h2>
                    <p>Inference mode is labeled in the response.</p>
                  </div>
                  <span className="tag neutral">
                    {active ? "ACTIVE MODEL" : "SIMULATED DEMO"}
                  </span>
                </div>
                <label className="field-label">PRODUCT DESCRIPTION</label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
                <button
                  className="primary run"
                  disabled={busy}
                  onClick={() =>
                    act(async () => {
                      const result = await post<Record<string, unknown>>(
                        `${API}/tasks/product-attributes/predict`,
                        { input_text: text },
                      );
                      setPrediction(result);
                      setCorrection(JSON.stringify(result.output, null, 2));
                    })
                  }
                >
                  <Play size={15} /> Run extraction
                </button>
                {prediction && (
                  <>
                    <div className="result-head">
                      <b>Structured output</b>
                      <span className="tag warn">
                        {String(prediction.mode).toUpperCase()}
                      </span>
                    </div>
                    <textarea
                      aria-label="Correct extracted attributes as JSON"
                      value={correction}
                      onChange={(e) => setCorrection(e.target.value)}
                    />
                    <button
                      className="primary"
                      disabled={busy}
                      onClick={() =>
                        act(async () => {
                          const corrected_output = JSON.parse(correction);
                          await post(`${API}/feedback`, {
                            input_text: text,
                            corrected_output,
                            original_prediction: prediction.output,
                            source: "dashboard",
                            notes: "Operator correction from playground",
                          });
                          setPrediction(null);
                          setPage("Corrections");
                        })
                      }
                    >
                      <Check size={14} /> Submit correction for review
                    </button>
                  </>
                )}
              </section>
              <section className="panel guidance">
                <div className="guidance-icon">
                  <Sparkles />
                </div>
                <h2>Evidence before deployment</h2>
                <p>
                  Simulation helps inspect the interface. It does not measure an
                  ML model or support a promotion decision.
                </p>
                <div className="checkline">
                  <Check size={14} /> Schema validated locally
                </div>
                <div className="checkline">
                  <Check size={14} /> Unknown attributes remain null
                </div>
                <div className="checkline">
                  <ArrowDownRight size={14} /> Model quality not measured
                </div>
              </section>
            </div>
          )}
          {["Experiments", "Evaluation", "Registry"].includes(page) && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2>
                    {page === "Experiments"
                      ? "Training experiments"
                      : page === "Evaluation"
                        ? "Evaluation evidence"
                        : "Local model registry"}
                  </h2>
                  <p>
                    {page === "Evaluation"
                      ? "No synthetic performance claims are generated. Reports require real predictions."
                      : "Persistent backend state · hardware-dependent steps require MLX and model weights."}
                  </p>
                </div>
                {page === "Registry" && (
                  <button className="quiet" onClick={refresh}>
                    <RefreshCw size={14} /> Refresh
                  </button>
                )}
              </div>
              {page === "Registry" && (
                <>
                  <div className="active-banner">
                    <ShieldCheck size={18} />
                    <div>
                      <b>Active deployment</b>
                      <small>{active?.id || "No model is deployed"}</small>
                    </div>
                    <span className="tag neutral">
                      {active?.status || "EMPTY"}
                    </span>
                  </div>
                  {active && (
                    <button
                      className="quiet"
                      disabled={busy}
                      onClick={() => {
                        if (
                          window.confirm(
                            "Roll back to the previous healthy deployment?",
                          )
                        )
                          void act(() =>
                            post(`${API}/models/rollback`, {
                              reason:
                                "Operator initiated rollback from dashboard.",
                            }),
                          );
                      }}
                    >
                      Rollback deployment
                    </button>
                  )}
                  {models.length ? (
                    models.map((m) => (
                      <div className="model-row" key={m.id}>
                        <Boxes size={18} />
                        <div>
                          <b>{m.id}</b>
                          <small>
                            {m.base_model} · {m.status}
                          </small>
                        </div>
                        <span className="tag neutral">
                          {m.status.toUpperCase()}
                        </span>
                        {m.status === "evaluated" && (
                          <button
                            className="approve"
                            disabled={busy}
                            onClick={() => {
                              if (
                                window.confirm(
                                  "Promote this candidate after running its readiness checks?",
                                )
                              )
                                void act(() =>
                                  post(`${API}/models/${m.id}/promote`, {
                                    reason:
                                      "Operator approved after reviewing measured evaluation.",
                                  }),
                                );
                            }}
                          >
                            Promote
                          </button>
                        )}
                      </div>
                    ))
                  ) : (
                    <Empty
                      icon={<Boxes />}
                      title="No registered models"
                      detail="Train on an MLX-capable Apple Silicon machine to register an adapter."
                    />
                  )}
                </>
              )}
              {page === "Registry" && (
                <>
                  <h3 className="section-label">DEPLOYMENT HISTORY</h3>
                  {deploymentHistory.length ? (
                    deploymentHistory.slice(0, 12).map((entry) => (
                      <article className="dataset" key={entry.id}>
                        <div className="dataset-main">
                          <b>
                            {entry.action} · {entry.model_id}
                          </b>
                          <small>
                            {entry.operator || "operator"} ·{" "}
                            {new Date(entry.created_at).toLocaleString()} ·{" "}
                            {entry.reason || "No reason recorded"}
                          </small>
                          {Array.isArray(entry.details.failed_checks) &&
                            entry.details.failed_checks.length > 0 && (
                              <small className="tag warn">
                                Override checks failed:{" "}
                                {entry.details.failed_checks.join(", ")}
                              </small>
                            )}
                        </div>
                      </article>
                    ))
                  ) : (
                    <p className="muted">No deployment actions recorded.</p>
                  )}
                </>
              )}
              {page === "Experiments" && (
                <>
                  <div className="actions">
                    <select
                      className="search"
                      value={selectedDataset}
                      onChange={(e) => setSelectedDataset(e.target.value)}
                    >
                      <option value="">Choose dataset version</option>
                      {datasets.map((d) => (
                        <option value={d.id} key={d.id}>
                          {d.id} · {d.manifest.counts.train} train
                        </option>
                      ))}
                    </select>
                    <button
                      className="primary"
                      disabled={busy || !selectedDataset}
                      onClick={() =>
                        act(() =>
                          post(`${API}/train/jobs`, {
                            dataset_id: selectedDataset,
                            preset: "smoke",
                          }),
                        )
                      }
                    >
                      <FlaskConical size={14} /> Start smoke LoRA job
                    </button>
                  </div>
                  {experiments.length ? (
                    experiments.map((job) => (
                      <article className="dataset" key={job.id}>
                        <div className="dataset-icon">
                          <FlaskConical size={17} />
                        </div>
                        <div className="dataset-main">
                          <b>{job.id}</b>
                          <small>
                            {job.status.toUpperCase()} ·{" "}
                            {String(job.config.model || "model")} · dataset{" "}
                            {job.dataset_id}
                          </small>
                          <small>
                            {String(
                              job.manifest.log || "Worker log initializing",
                            )}
                          </small>
                        </div>
                        <span
                          className={
                            "tag " +
                            (job.status === "completed"
                              ? "good"
                              : job.status === "failed"
                                ? "danger"
                                : "neutral")
                          }
                        >
                          {job.status.toUpperCase()}
                        </span>
                        {job.status === "running" && (
                          <button
                            className="reject"
                            onClick={() => {
                              if (
                                window.confirm(
                                  "Cancel this local training process?",
                                )
                              )
                                void act(() =>
                                  post(
                                    `${API}/train/jobs/${job.id}/cancel`,
                                    {},
                                  ),
                                );
                            }}
                          >
                            Cancel
                          </button>
                        )}
                      </article>
                    ))
                  ) : (
                    <Empty
                      icon={<FlaskConical />}
                      title="No training jobs"
                      detail="Training requires an Apple Silicon Mac with MLX and local model weights."
                    />
                  )}
                </>
              )}
              {page === "Evaluation" && (
                <EvaluationPanel
                  models={models}
                  evaluations={evaluations}
                  evaluationSets={evaluationSets}
                  busy={busy}
                  run={async (modelId, evaluationSetId) => {
                    const model = models.find((x) => x.id === modelId);
                    if (!model?.dataset_id)
                      throw new Error("Candidate has no dataset lineage");
                    return post(`${API}/evaluations`, {
                      dataset_id: model.dataset_id,
                      candidate_id: modelId,
                      evaluation_set_id: evaluationSetId || null,
                    });
                  }}
                  act={act}
                />
              )}
            </section>
          )}
          <footer>
            <span>
              LLM FEEDBACK PIPELINE <b>·</b> LOCAL MODEL OPERATIONS
            </span>
            <span>
              <i style={{ background: connected ? "#75c889" : "#de7771" }} />{" "}
              {connected ? "DATABASE CONNECTED" : "API DISCONNECTED"} ·{" "}
              {new Date().toLocaleTimeString()}
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}
function Metric({
  label,
  value,
  icon,
  detail,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  detail: string;
}) {
  return (
    <article className="metric">
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-icon">{icon}</span>
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}
function Empty({
  icon,
  title,
  detail,
}: {
  icon: React.ReactNode;
  title: string;
  detail: string;
}) {
  return (
    <div className="empty">
      <div>{icon}</div>
      <b>{title}</b>
      <p>{detail}</p>
    </div>
  );
}

const root = document.getElementById("root");
if (root)
  createRoot(root).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );

function EvaluationPanel({
  models,
  evaluations,
  evaluationSets,
  busy,
  run,
  act,
}: {
  models: Model[];
  evaluations: EvalRun[];
  evaluationSets: EvaluationSet[];
  busy: boolean;
  run: (id: string, evaluationSetId?: string) => Promise<unknown>;
  act: (fn: () => Promise<unknown>) => Promise<void>;
}) {
  const [selected, setSelected] = useState(""),
    [selectedSet, setSelectedSet] = useState(""),
    [regressionsOnly, setRegressionsOnly] = useState(true);
  const latest = evaluations[0]?.report;
  const examples = latest
    ? latest.candidate.examples
        .map((candidate: any, index: number) => ({
          candidate,
          baseline: latest.baseline.examples[index],
          outcome: latest.cases[index]?.outcome,
        }))
        .filter((row: any) => !regressionsOnly || row.outcome === "regressed")
    : [];
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>Paired model evaluation</h2>
          <p>
            Untuned, training-only prompting, and adapter results use the same
            frozen test examples.
          </p>
        </div>
        <div className="actions">
          <select
            className="search"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            <option value="">Choose evaluated candidate</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id} · {m.status}
              </option>
            ))}
          </select>
          <select
            className="search"
            value={selectedSet}
            onChange={(e) => setSelectedSet(e.target.value)}
          >
            <option value="">Use dataset test split</option>
            {evaluationSets.map((set) => (
              <option key={set.id} value={set.id}>
                {set.id} · n={set.manifest.count}
              </option>
            ))}
          </select>
          <button
            className="primary"
            disabled={busy || !selected}
            onClick={() => act(() => run(selected, selectedSet || undefined))}
          >
            <Activity size={14} /> Run real evaluation
          </button>
        </div>
      </div>
      {latest ? (
        <>
          <div className="metrics">
            <Metric
              label="BASE EXACT MATCH"
              value={
                Math.round(Number(latest.baseline.exact_match_accuracy) * 100) +
                "%"
              }
              icon={<Layers3 />}
              detail={latest.baseline_id || "Untuned model"}
            />
            <Metric
              label="CANDIDATE EXACT MATCH"
              value={
                Math.round(
                  Number(latest.candidate.exact_match_accuracy) * 100,
                ) + "%"
              }
              icon={<ArrowUpRight />}
              detail={latest.candidate_id}
            />
            <Metric
              label="IMPROVED / REGRESSED"
              value={latest.improved + " / " + latest.regressed}
              icon={<Activity />}
              detail={"n = " + latest.candidate.n}
            />
            <Metric
              label="SCHEMA ADHERENCE"
              value={
                Math.round(Number(latest.candidate.schema_adherence) * 100) +
                "%"
              }
              icon={<ShieldCheck />}
              detail="Complete JSON schema outputs"
            />
          </div>
          {latest.sample_warning && (
            <div className="notice">{latest.sample_warning}</div>
          )}
          {latest.paired_statistics && (
            <div className="notice">
              Paired bootstrap 95% interval for accuracy change:{" "}
              {Math.round(
                latest.paired_statistics.confidence_interval[0] * 100,
              )}
              % to{" "}
              {Math.round(
                latest.paired_statistics.confidence_interval[1] * 100,
              )}
              % · n={latest.candidate.n} · seed {latest.paired_statistics.seed}
            </div>
          )}
          <h3 className="section-label">
            CAPABILITY COMPARISON · FIELD ACCURACY
          </h3>
          <div className="field-bars">
            {Object.entries(latest.candidate.per_field_accuracy || {}).map(
              ([name, value]) => (
                <div key={name}>
                  <span>{name}</span>
                  <div className="comparison-bars">
                    <i
                      title={
                        "Untuned " +
                        Math.round(
                          Number(latest.baseline.per_field_accuracy?.[name]) *
                            100,
                        ) +
                        "%"
                      }
                      style={{
                        width:
                          Math.max(
                            0,
                            Math.min(
                              100,
                              Number(
                                latest.baseline.per_field_accuracy?.[name],
                              ) * 100,
                            ),
                          ) + "%",
                      }}
                    />
                    <i
                      title={"Adapted " + Math.round(Number(value) * 100) + "%"}
                      style={{
                        width:
                          Math.max(0, Math.min(100, Number(value) * 100)) + "%",
                      }}
                    />
                  </div>
                  <b>
                    {Math.round(
                      Number(latest.baseline.per_field_accuracy?.[name]) * 100,
                    )}{" "}
                    / {Math.round(Number(value) * 100)}%
                  </b>
                </div>
              ),
            )}
          </div>
          <small className="muted">
            Bar pair: untuned baseline / adapted candidate.
          </small>
          <h3 className="section-label">IMPROVEMENTS AND REGRESSIONS</h3>
          <div className="metrics">
            <Metric
              label="IMPROVED"
              value={String(latest.improved)}
              icon={<ArrowUpRight />}
              detail="Baseline wrong, candidate correct"
            />
            <Metric
              label="REGRESSED"
              value={String(latest.regressed)}
              icon={<TriangleAlert />}
              detail="Baseline correct, candidate wrong"
            />
            <Metric
              label="UNCHANGED CORRECT"
              value={String(latest.both_correct)}
              icon={<Check />}
              detail="Both models correct"
            />
            <Metric
              label="UNCHANGED INCORRECT"
              value={String(latest.both_wrong)}
              icon={<X />}
              detail="Both models incorrect"
            />
          </div>
          <h3 className="section-label">ACCURACY AND LATENCY</h3>
          <div className="dataset-list">
            {[
              ["Untuned", latest.baseline],
              ["Prompting only", latest.prompting_only],
              ["Adapted", latest.candidate],
            ]
              .filter((item: any) => item[1])
              .map(([label, metrics]: any) => (
                <article className="dataset" key={label}>
                  <div className="dataset-main">
                    <b>{label}</b>
                    <small>
                      Exact match{" "}
                      {Math.round(Number(metrics.exact_match_accuracy) * 100)}%
                      · median warm latency{" "}
                      {metrics.latency_ms_median === undefined
                        ? "—"
                        : Math.round(metrics.latency_ms_median) + " ms"}{" "}
                      · cold load{" "}
                      {metrics.cold_load_ms === undefined
                        ? "—"
                        : Math.round(metrics.cold_load_ms) + " ms"}
                    </small>
                    <small>
                      {metrics.strategy ||
                        "Greedy decoding · sequential evaluation"}
                    </small>
                  </div>
                </article>
              ))}
          </div>
          <h3 className="section-label">HELD-OUT EXAMPLE INSPECTION</h3>
          <label className="checkline">
            <input
              type="checkbox"
              checked={regressionsOnly}
              onChange={(e) => setRegressionsOnly(e.target.checked)}
            />{" "}
            Show regressions only
          </label>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>INPUT</th>
                  <th>GROUND TRUTH</th>
                  <th>UNTUNED</th>
                  <th>ADAPTED</th>
                  <th>OUTCOME / ERROR</th>
                </tr>
              </thead>
              <tbody>
                {examples.slice(0, 100).map((row: any, index: number) => (
                  <tr key={index}>
                    <td>{row.candidate.input_text || "Held-out example"}</td>
                    <td>
                      <code>{JSON.stringify(row.candidate.ground_truth)}</code>
                    </td>
                    <td>
                      <code>
                        {JSON.stringify(row.baseline.prediction) ||
                          "Invalid output"}
                      </code>
                    </td>
                    <td>
                      <code>
                        {JSON.stringify(row.candidate.prediction) ||
                          "Invalid output"}
                      </code>
                    </td>
                    <td>
                      {row.outcome}
                      <small>
                        {(row.candidate.error_categories || []).join(", ") ||
                          "—"}
                      </small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {examples.length === 0 && (
              <Empty
                icon={<Check />}
                title="No matching examples"
                detail={
                  regressionsOnly
                    ? "No regressions in this evaluation."
                    : "No examples available."
                }
              />
            )}
          </div>
          <div className="notice">
            Power draw and cost per correct task are unavailable until the
            operator supplies local measurements and cost assumptions.
            Evaluation process peak RSS:{" "}
            {latest.observed_process_peak_rss_bytes
              ? Math.round(
                  latest.observed_process_peak_rss_bytes / 1024 / 1024,
                ) + " MiB"
              : "unavailable"}
            .
          </div>
        </>
      ) : (
        <Empty
          icon={<Activity />}
          title="No evaluation runs"
          detail="Choose a trained candidate to run real paired inference. No synthetic scores are shown."
        />
      )}
    </section>
  );
}
