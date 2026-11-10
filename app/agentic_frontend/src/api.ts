
export type Insight = {
  title: string;
  finding: string;
  business_impact: string;
  recommendation: string;
  priority: "high" | "medium" | "low";
  caveats: string[];
  evidence: {
    execution_index: number;
    fact: string;
  }[];
};

export type Investigation = {
  execution_index: number;
  tool_name: string;
  target_columns: string[];
  success: boolean;
  result: unknown;
  error?: string | null;
};

export type MetricProfile = {
  column_name: string;
  role: string;
  confidence: number;
  signals: string[];
};

export type Hypothesis = {
  hypothesis: string;
  supporting_execution_indexes?: number[];
  status?: string;
  conclusion?: string | null;
};

export type AnalysisResponse = {
  status: string;
  dataset: {
    name: string;
    rows: number;
    columns: number;
  };
  tools_available: number;
  coverage: number;
  investigated_columns: string[];
  metric_profiles: MetricProfile[];
  investigations: Investigation[];
  insights: Insight[];
  hypotheses: Hypothesis[];
};

export type PostgresConnectionResponse = {
  success: boolean;
  message: string;
};

export type PostgresTablesResponse = {
  schema: string;
  tables: string[];
  count: number;
};

export type PostgresPreviewResponse = {
  schema: string;
  table: string;
  rows_returned: number;
  columns: string[];
  rows: Record<string, unknown>[];
};

export type PostgresAnalyzeRequest = {
  schema_name?: string;
  table_name: string;
  row_limit?: number;
};

// API base URL
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// Shared request handler
async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error(
      `Cannot connect to the backend at ${API_BASE_URL}. Make sure FastAPI is running.`
    );
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : `Request failed with status ${response.status}.`;

    throw new Error(message);
  }

  return body as T;
}

// --------------------------------------------------
// Health check
// --------------------------------------------------

export type BackendHealthResponse = {
  status: string;
  service?: string;
};

export function testBackendConnection(): Promise<BackendHealthResponse> {
  return request<BackendHealthResponse>("/api/health");
}

// --------------------------------------------------
// CSV analysis
// --------------------------------------------------

export async function analyzeDataset(
  file: File
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<AnalysisResponse>("/api/analyze", {
    method: "POST",
    body: formData,
  });
}

// --------------------------------------------------
// PostgreSQL connection test
// --------------------------------------------------

export function testPostgresConnection(): Promise<PostgresConnectionResponse> {
  return request<PostgresConnectionResponse>(
    "/api/postgres/test-connection"
  );
}

// --------------------------------------------------
// PostgreSQL table listing
// --------------------------------------------------

export function getPostgresTables(
  schemaName = "public"
): Promise<PostgresTablesResponse> {
  const params = new URLSearchParams({
    schema_name: schemaName,
  });

  return request<PostgresTablesResponse>(
    `/api/postgres/tables?${params.toString()}`
  );
}

// --------------------------------------------------
// PostgreSQL table preview
// --------------------------------------------------

export function previewPostgresTable(
  tableName: string,
  schemaName = "public",
  limit = 100
): Promise<PostgresPreviewResponse> {
  const params = new URLSearchParams({
    table_name: tableName,
    schema_name: schemaName,
    limit: String(limit),
  });

  return request<PostgresPreviewResponse>(
    `/api/postgres/preview?${params.toString()}`
  );
}

// --------------------------------------------------
// PostgreSQL table analysis
// --------------------------------------------------

export function analyzePostgresTable(
  input: PostgresAnalyzeRequest
): Promise<AnalysisResponse> {
  return request<AnalysisResponse>("/api/postgres/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      schema_name: input.schema_name ?? "public",
      table_name: input.table_name,
      row_limit: input.row_limit ?? 50000,
    }),
  });
}
