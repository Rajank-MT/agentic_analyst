export type Insight = {
  title: string;
  finding: string;
  business_impact: string;
  recommendation: string;
  priority: 'high' | 'medium' | 'low';
  caveats: string[];
  evidence: { execution_index: number; fact: string }[];
};

export type Investigation = {
  execution_index: number;
  tool_name: string;
  target_columns: string[];
  success: boolean;
  result: unknown;
  error?: string | null;
};

export type AnalysisResponse = {
  status: string;
  dataset: { name: string; rows: number; columns: number };
  tools_available: number;
  coverage: number;
  investigated_columns: string[];
  metric_profiles: { column_name: string; role: string; confidence: number; signals: string[] }[];
  investigations: Investigation[];
  insights: Insight[];
  hypotheses: unknown[];
};

export async function analyzeDataset(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:8000/api/analyze', {
    method: 'POST',
    body: formData,
  });

  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || 'Analysis failed.');
  }
  return body;
}
