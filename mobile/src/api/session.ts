import { apiFetch } from './client';

export interface PlanRequest {
  template: string;
  prompt?: string;
}

export interface Instruction {
  timestamp_offset: number;
  combo_display_name: string;
  actions: string[];
}

export interface Round {
  round_number: number;
  duration_seconds: number;
  rest_after_seconds: number;
  instructions: Instruction[];
}

export interface PlanDetail {
  session_type: string;
  template: string;
  focus: string;
  total_duration_minutes: number;
  rounds: Round[];
  pace_interval_sec: [number, number];
}

export interface PlanResponse {
  id: string;
  llm_model: string;
  plan: PlanDetail;
}

export interface SessionSummary {
  id: string;
  drill_plan_id: string;
  template_name: string;
  started_at: string;
  completed_at: string | null;
  total_duration_sec: number;
  rounds_completed: number;
  rounds_total: number;
  combos_delivered: number;
  status: string;
}

export interface SessionDetail extends SessionSummary {
  delivery_log_json: Record<string, unknown>;
}

export interface SessionLogRequest {
  drill_plan_id: string;
  template_name: string;
  started_at: string;
  completed_at: string;
  total_duration_sec: number;
  rounds_completed: number;
  rounds_total: number;
  combos_delivered: number;
  status: string;
}

export const generatePlan = (body: PlanRequest): Promise<PlanResponse> =>
  apiFetch<PlanResponse>('/api/sessions/plan', {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const logSession = (body: SessionLogRequest): Promise<SessionSummary> =>
  apiFetch<SessionSummary>('/api/sessions/log', {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const fetchSessions = (limit = 20, offset = 0): Promise<SessionSummary[]> =>
  apiFetch<SessionSummary[]>(`/api/sessions?limit=${limit}&offset=${offset}`);

export const fetchSession = (id: string): Promise<SessionDetail> =>
  apiFetch<SessionDetail>(`/api/sessions/${id}`);
