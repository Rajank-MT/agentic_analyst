import { useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Database,
  FileSpreadsheet,
  FlaskConical,
  Lightbulb,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  Target,
  Upload,
  XCircle,
  Zap,
} from "lucide-react";
import "./index.css";

const API_URL = "http://127.0.0.1:8000";

type MetricProfile = {
  column_name: string;
  role: string;
  confidence: number;
  signals?: string[];
};

type Investigation = {
  execution_index: number;
  tool_name: string;
  target_columns: string[];
  success: boolean;
  result: unknown;
  error?: string | null;
};

type DatasetInfo = {
  name: string;
  rows: number;
  columns: number;
};

type AnalysisResponse = {
  status: string;
  dataset: DatasetInfo;
  tools_available: number;
  coverage: number;
  investigated_columns: string[];
  metric_profiles: MetricProfile[];
  investigations: Investigation[];
  insights: unknown[];
  hypotheses: unknown[];
};

type Tab =
  | "overview"
  | "investigations"
  | "insights"
  | "hypotheses"
  | "schema";

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-IN").format(value);
}

function prettyLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function safeString(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function getInsightText(value: unknown): string {
  if (typeof value === "string") return value;

  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;

    const candidates = [
      object.insight,
      object.description,
      object.statement,
      object.finding,
      object.summary,
      object.text,
      object.title,
    ];

    const found = candidates.find(
      (item) => typeof item === "string" && item.trim().length > 0
    );

    if (found) return String(found);

    return safeString(value);
  }

  return String(value);
}

function getHypothesisText(value: unknown): string {
  if (typeof value === "string") return value;

  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;

    const candidates = [
      object.hypothesis,
      object.statement,
      object.description,
      object.text,
      object.title,
    ];

    const found = candidates.find(
      (item) => typeof item === "string" && item.trim().length > 0
    );

    if (found) return String(found);

    return safeString(value);
  }

  return String(value);
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [expandedInvestigation, setExpandedInvestigation] = useState<
    number | null
  >(null);

  const successfulInvestigations = useMemo(
    () =>
      data?.investigations.filter((item) => item.success).length ?? 0,
    [data]
  );

  const failedInvestigations = useMemo(
    () =>
      data?.investigations.filter((item) => !item.success).length ?? 0,
    [data]
  );

  function handleFile(selectedFile: File | null) {
    setError("");

    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file.");
      return;
    }

    setFile(selectedFile);
    setData(null);
    setActiveTab("overview");
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);

    const droppedFile = event.dataTransfer.files?.[0];

    if (droppedFile) {
      handleFile(droppedFile);
    }
  }

  async function runAnalysis() {
    if (!file) {
      setError("Please select a CSV dataset first.");
      return;
    }

    setLoading(true);
    setError("");
    setData(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const responseText = await response.text();

      let responseData: any;

      try {
        responseData = JSON.parse(responseText);
      } catch {
        throw new Error(
          `Backend returned an invalid response (${response.status}).`
        );
      }

      if (!response.ok) {
        throw new Error(
          responseData?.detail ||
            `Analysis failed with status ${response.status}.`
        );
      }

      setData(responseData as AnalysisResponse);
      setActiveTab("overview");
    } catch (err) {
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setError(
          "Cannot connect to the FastAPI backend. Make sure it is running on http://127.0.0.1:8000."
        );
      } else {
        setError(
          err instanceof Error ? err.message : "Something went wrong."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  function resetAnalysis() {
    setFile(null);
    setData(null);
    setError("");
    setActiveTab("overview");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={22} />
          </div>

          <div>
            <div className="brand-name">Agentic Data Analyst</div>
            <div className="brand-subtitle">
              Autonomous intelligence for your data
            </div>
          </div>
        </div>

        <div className="topbar-status">
          <span className="status-dot" />
          AI ENGINE READY
        </div>
      </header>

      <main className="main-content">
        {!data ? (
          <UploadScreen
            file={file}
            loading={loading}
            dragging={dragging}
            error={error}
            onFile={handleFile}
            onDrop={handleDrop}
            onDragEnter={() => setDragging(true)}
            onDragLeave={() => setDragging(false)}
            onAnalyze={runAnalysis}
          />
        ) : (
          <Dashboard
            data={data}
            file={file}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            expandedInvestigation={expandedInvestigation}
            setExpandedInvestigation={setExpandedInvestigation}
            successfulInvestigations={successfulInvestigations}
            failedInvestigations={failedInvestigations}
            onReset={resetAnalysis}
          />
        )}
      </main>
    </div>
  );
}

function UploadScreen({
  file,
  loading,
  dragging,
  error,
  onFile,
  onDrop,
  onDragEnter,
  onDragLeave,
  onAnalyze,
}: {
  file: File | null;
  loading: boolean;
  dragging: boolean;
  error: string;
  onFile: (file: File | null) => void;
  onDrop: (event: React.DragEvent<HTMLDivElement>) => void;
  onDragEnter: () => void;
  onDragLeave: () => void;
  onAnalyze: () => void;
}) {
  return (
    <section className="upload-page">
      <div className="hero-section">
        <div className="eyebrow">
          <Sparkles size={15} />
          AGENTIC ANALYTICS PLATFORM
        </div>

        <h1>
          Let your data
          <br />
          <span>think for itself.</span>
        </h1>

        <p>
          Upload a dataset and let the AI agent discover patterns,
          investigate relationships, validate hypotheses, and surface
          evidence-backed business insights automatically.
        </p>
      </div>

      <div
        className={`upload-card ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => event.preventDefault()}
        onDragEnter={(event) => {
          event.preventDefault();
          onDragEnter();
        }}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div className="upload-icon">
          {loading ? (
            <Loader2 className="spin" size={32} />
          ) : (
            <Upload size={32} />
          )}
        </div>

        <h2>
          {loading
            ? "Agent is investigating your dataset..."
            : file
            ? file.name
            : "Drop your CSV dataset here"}
        </h2>

        <p>
          {loading
            ? "The agent is selecting analysis tools and evaluating evidence."
            : "or choose a file from your computer"}
        </p>

        {!loading && (
          <label className="choose-file">
            <FileSpreadsheet size={17} />
            Choose CSV
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) =>
                onFile(event.target.files?.[0] ?? null)
              }
            />
          </label>
        )}

        {file && !loading && (
          <div className="selected-file">
            <FileSpreadsheet size={18} />

            <div className="selected-file-info">
              <strong>{file.name}</strong>
              <span>{(file.size / 1024).toFixed(1)} KB</span>
            </div>

            <button
              className="remove-file"
              onClick={() => onFile(null)}
              title="Remove file"
            >
              <XCircle size={19} />
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner">
          <XCircle size={19} />
          <span>{error}</span>
        </div>
      )}

      <button
        className="analyze-button"
        disabled={!file || loading}
        onClick={onAnalyze}
      >
        {loading ? (
          <>
            <Loader2 className="spin" size={19} />
            Running Autonomous Investigation
          </>
        ) : (
          <>
            <Play size={18} fill="currentColor" />
            Start Autonomous Analysis
          </>
        )}
      </button>

      <div className="feature-row">
        <Feature
          icon={<Database size={18} />}
          title="Dataset Profiling"
          text="Understand your data automatically"
        />

        <Feature
          icon={<BrainCircuit size={18} />}
          title="Agent Reasoning"
          text="Choose what to investigate next"
        />

        <Feature
          icon={<FlaskConical size={18} />}
          title="Evidence Validation"
          text="Support insights with statistics"
        />
      </div>
    </section>
  );
}

function Feature({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="feature">
      <div className="feature-icon">{icon}</div>
      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}

function Dashboard({
  data,
  file,
  activeTab,
  setActiveTab,
  expandedInvestigation,
  setExpandedInvestigation,
  successfulInvestigations,
  failedInvestigations,
  onReset,
}: {
  data: AnalysisResponse;
  file: File | null;
  activeTab: Tab;
  setActiveTab: (tab: Tab) => void;
  expandedInvestigation: number | null;
  setExpandedInvestigation: (index: number | null) => void;
  successfulInvestigations: number;
  failedInvestigations: number;
  onReset: () => void;
}) {
  return (
    <>
      <div className="dashboard-header">
        <div>
          <div className="breadcrumb">
            <span>Workspace</span>
            <ChevronRight size={14} />
            <span>Analysis</span>
          </div>

          <h1>{data.dataset.name}</h1>

          <p>
            Autonomous investigation completed with{" "}
            {data.investigations.length} analysis steps.
          </p>
        </div>

        <button className="new-analysis" onClick={onReset}>
          <RefreshCw size={17} />
          New Analysis
        </button>
      </div>

      <div className="stats-grid">
        <StatCard
          icon={<Database />}
          label="Rows"
          value={formatNumber(data.dataset.rows)}
          detail="Dataset records"
        />

        <StatCard
          icon={<BarChart3 />}
          label="Columns"
          value={formatNumber(data.dataset.columns)}
          detail="Detected features"
        />

        <StatCard
          icon={<Zap />}
          label="Investigations"
          value={formatNumber(data.investigations.length)}
          detail={`${successfulInvestigations} successful`}
        />

        <StatCard
          icon={<Target />}
          label="Coverage"
          value={`${data.coverage}%`}
          detail="Columns investigated"
        />
      </div>

      <div className="dashboard-layout">
        <aside className="sidebar">
          <div className="sidebar-label">ANALYSIS</div>

          <NavButton
            active={activeTab === "overview"}
            icon={<Activity size={18} />}
            label="Overview"
            count={undefined}
            onClick={() => setActiveTab("overview")}
          />

          <NavButton
            active={activeTab === "investigations"}
            icon={<Search size={18} />}
            label="Investigations"
            count={data.investigations.length}
            onClick={() => setActiveTab("investigations")}
          />

          <NavButton
            active={activeTab === "insights"}
            icon={<Lightbulb size={18} />}
            label="Business Insights"
            count={data.insights.length}
            onClick={() => setActiveTab("insights")}
          />

          <NavButton
            active={activeTab === "hypotheses"}
            icon={<FlaskConical size={18} />}
            label="Hypotheses"
            count={data.hypotheses.length}
            onClick={() => setActiveTab("hypotheses")}
          />

          <NavButton
            active={activeTab === "schema"}
            icon={<Database size={18} />}
            label="Data Schema"
            count={data.metric_profiles.length}
            onClick={() => setActiveTab("schema")}
          />

          <div className="sidebar-divider" />

          <div className="agent-status-card">
            <div className="agent-status-top">
              <div className="mini-agent-icon">
                <BrainCircuit size={17} />
              </div>

              <span className="live-pill">
                <span />
                COMPLETE
              </span>
            </div>

            <strong>Agent Investigation</strong>

            <p>
              The agent autonomously selected analysis directions and
              evaluated the results.
            </p>

            {failedInvestigations > 0 && (
              <div className="sidebar-warning">
                {failedInvestigations} investigation
                {failedInvestigations > 1 ? "s" : ""} failed
              </div>
            )}
          </div>
        </aside>

        <section className="content-panel">
          {activeTab === "overview" && (
            <Overview
              data={data}
              successfulInvestigations={successfulInvestigations}
            />
          )}

          {activeTab === "investigations" && (
            <Investigations
              data={data}
              expandedInvestigation={expandedInvestigation}
              setExpandedInvestigation={setExpandedInvestigation}
            />
          )}

          {activeTab === "insights" && <Insights data={data} />}

          {activeTab === "hypotheses" && <Hypotheses data={data} />}

          {activeTab === "schema" && <Schema data={data} />}
        </section>
      </div>

      {file && (
        <div className="footer-status">
          <span>
            <CheckCircle2 size={15} />
            Analysis generated from {file.name}
          </span>

          <span>Agentic Data Analyst v1.0</span>
        </div>
      )}
    </>
  );
}

function StatCard({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>

      <div className="stat-content">
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function NavButton({
  active,
  icon,
  label,
  count,
  onClick,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  count: number | undefined;
  onClick: () => void;
}) {
  return (
    <button className={`nav-button ${active ? "active" : ""}`} onClick={onClick}>
      {icon}
      <span>{label}</span>

      {count !== undefined && <b>{count}</b>}
    </button>
  );
}

function Overview({
  data,
  successfulInvestigations,
}: {
  data: AnalysisResponse;
  successfulInvestigations: number;
}) {
  return (
    <div className="tab-content">
      <SectionHeading
        eyebrow="ANALYSIS OVERVIEW"
        title="What the agent discovered"
        description="A high-level view of the autonomous investigation."
      />

      <div className="overview-grid">
        <div className="panel-card coverage-card">
          <div className="card-heading">
            <div>
              <span className="card-eyebrow">INVESTIGATION COVERAGE</span>
              <h3>Dataset exploration</h3>
            </div>

            <Target size={20} />
          </div>

          <div className="coverage-display">
            <div
              className="coverage-ring"
              style={{
                background: `conic-gradient(
                  #8b5cf6 ${data.coverage * 3.6}deg,
                  rgba(255,255,255,.07) ${data.coverage * 3.6}deg
                )`,
              }}
            >
              <div>
                <strong>{data.coverage}%</strong>
                <span>covered</span>
              </div>
            </div>

            <div className="coverage-copy">
              <strong>
                {data.investigated_columns.length} of{" "}
                {data.metric_profiles.filter(
                  (profile) => profile.role !== "entity"
                ).length}{" "}
                analytical columns
              </strong>

              <p>
                The agent explored the dataset across multiple analysis
                directions rather than repeatedly focusing on a single metric.
              </p>
            </div>
          </div>
        </div>

        <div className="panel-card">
          <div className="card-heading">
            <div>
              <span className="card-eyebrow">AGENT ACTIVITY</span>
              <h3>Investigation execution</h3>
            </div>

            <BrainCircuit size={20} />
          </div>

          <div className="activity-list">
            <ActivityRow
              label="Analysis steps"
              value={data.investigations.length}
            />

            <ActivityRow
              label="Successful executions"
              value={successfulInvestigations}
              success
            />

            <ActivityRow
              label="Available tools"
              value={data.tools_available}
            />

            <ActivityRow
              label="Insights generated"
              value={data.insights.length}
            />

            <ActivityRow
              label="Hypotheses tracked"
              value={data.hypotheses.length}
            />
          </div>
        </div>
      </div>

      <div className="panel-card recent-card">
        <div className="card-heading">
          <div>
            <span className="card-eyebrow">RECENT ACTIVITY</span>
            <h3>Agent investigation trail</h3>
          </div>

          <Activity size={20} />
        </div>

        {data.investigations.length === 0 ? (
          <EmptyState text="No investigations were recorded." />
        ) : (
          <div className="timeline">
            {data.investigations.slice(0, 6).map((investigation) => (
              <div
                className="timeline-item"
                key={investigation.execution_index}
              >
                <div
                  className={`timeline-dot ${
                    investigation.success ? "success" : "failed"
                  }`}
                >
                  {investigation.success ? (
                    <CheckCircle2 size={14} />
                  ) : (
                    <XCircle size={14} />
                  )}
                </div>

                <div className="timeline-body">
                  <strong>{prettyLabel(investigation.tool_name)}</strong>

                  <span>
                    {investigation.target_columns?.length
                      ? investigation.target_columns.join(" • ")
                      : "Dataset-level analysis"}
                  </span>
                </div>

                <small>Step {investigation.execution_index + 1}</small>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ActivityRow({
  label,
  value,
  success,
}: {
  label: string;
  value: number;
  success?: boolean;
}) {
  return (
    <div className="activity-row">
      <span>{label}</span>
      <strong className={success ? "success-text" : ""}>{value}</strong>
    </div>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <span>{eyebrow}</span>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
}

function Investigations({
  data,
  expandedInvestigation,
  setExpandedInvestigation,
}: {
  data: AnalysisResponse;
  expandedInvestigation: number | null;
  setExpandedInvestigation: (index: number | null) => void;
}) {
  return (
    <div className="tab-content">
      <SectionHeading
        eyebrow="AGENT REASONING"
        title="Investigation trail"
        description="Every deterministic analysis executed by the autonomous agent."
      />

      <div className="investigation-list">
        {data.investigations.map((investigation) => {
          const expanded =
            expandedInvestigation === investigation.execution_index;

          return (
            <div
              className={`investigation-card ${
                expanded ? "expanded" : ""
              }`}
              key={investigation.execution_index}
            >
              <button
                className="investigation-header"
                onClick={() =>
                  setExpandedInvestigation(
                    expanded ? null : investigation.execution_index
                  )
                }
              >
                <div className="step-number">
                  {investigation.execution_index + 1}
                </div>

                <div className="investigation-main">
                  <div className="investigation-title">
                    <strong>
                      {prettyLabel(investigation.tool_name)}
                    </strong>

                    {investigation.success ? (
                      <span className="success-badge">
                        <CheckCircle2 size={13} />
                        SUCCESS
                      </span>
                    ) : (
                      <span className="failed-badge">
                        <XCircle size={13} />
                        FAILED
                      </span>
                    )}
                  </div>

                  <div className="column-tags">
                    {investigation.target_columns?.length ? (
                      investigation.target_columns.map((column) => (
                        <span key={column}>{column}</span>
                      ))
                    ) : (
                      <span>Dataset level</span>
                    )}
                  </div>
                </div>

                <div className="expand-icon">
                  {expanded ? (
                    <ChevronDown size={20} />
                  ) : (
                    <ChevronRight size={20} />
                  )}
                </div>
              </button>

              {expanded && (
                <div className="investigation-result">
                  {investigation.error && (
                    <div className="result-error">
                      <XCircle size={17} />
                      {investigation.error}
                    </div>
                  )}

                  <div className="result-label">DETERMINISTIC RESULT</div>

                  <pre>
                    {safeString(investigation.result) ||
                      "No result returned."}
                  </pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Insights({ data }: { data: AnalysisResponse }) {
  return (
    <div className="tab-content">
      <SectionHeading
        eyebrow="BUSINESS INTELLIGENCE"
        title="Evidence-backed insights"
        description="Meaningful findings generated from the analysis results."
      />

      {data.insights.length === 0 ? (
        <EmptyState text="No business insights were generated." />
      ) : (
        <div className="insight-grid">
          {data.insights.map((insight, index) => (
            <div className="insight-card" key={index}>
              <div className="insight-number">
                {String(index + 1).padStart(2, "0")}
              </div>

              <div className="insight-icon">
                <Lightbulb size={19} />
              </div>

              <div className="insight-body">
                <span className="card-eyebrow">KEY FINDING</span>

                <p>{getInsightText(insight)}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Hypotheses({ data }: { data: AnalysisResponse }) {
  return (
    <div className="tab-content">
      <SectionHeading
        eyebrow="HYPOTHESIS ENGINE"
        title="What the agent investigated"
        description="Hypotheses tracked during the autonomous investigation process."
      />

      {data.hypotheses.length === 0 ? (
        <EmptyState text="No hypotheses were recorded." />
      ) : (
        <div className="hypothesis-list">
          {data.hypotheses.map((hypothesis, index) => (
            <div className="hypothesis-card" key={index}>
              <div className="hypothesis-index">{index + 1}</div>

              <div className="hypothesis-content">
                <span>HYPOTHESIS {index + 1}</span>
                <p>{getHypothesisText(hypothesis)}</p>
              </div>

              <FlaskConical size={20} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Schema({ data }: { data: AnalysisResponse }) {
  return (
    <div className="tab-content">
      <SectionHeading
        eyebrow="DATA UNDERSTANDING"
        title="Metric profiles"
        description="How the agent classified and understood each dataset column."
      />

      <div className="schema-table-wrapper">
        <table className="schema-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Semantic Role</th>
              <th>Confidence</th>
              <th>Signals</th>
            </tr>
          </thead>

          <tbody>
            {data.metric_profiles.map((profile) => (
              <tr key={profile.column_name}>
                <td>
                  <div className="column-name">
                    <Database size={15} />
                    {profile.column_name}
                  </div>
                </td>

                <td>
                  <span className={`role-badge role-${profile.role}`}>
                    {prettyLabel(profile.role)}
                  </span>
                </td>

                <td>
                  <div className="confidence">
                    <div className="confidence-bar">
                      <div
                        style={{
                          width: `${Math.min(
                            Math.max(profile.confidence * 100, 0),
                            100
                          )}%`,
                        }}
                      />
                    </div>

                    <span>
                      {(profile.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>

                <td>
                  <div className="signals">
                    {profile.signals?.length
                      ? profile.signals.map((signal) => (
                          <span key={signal}>{signal}</span>
                        ))
                      : "—"}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      <Database size={25} />
      <p>{text}</p>
    </div>
  );
}

export default App;