import { apiFetch } from './client';

export interface PlanRequest {
  level?: 'beginner' | 'intermediate' | 'advanced';
  rounds?: number;
  round_duration_sec?: number;
  rest_sec?: number;
  program_day_id?: string;
}

export interface Chunk {
  text: string;        // chunk text e.g. "원투"
  clip_url: string;    // /audio/chunks/원투_1.mp3
  duration_ms: number;
}

export interface Segment {
  text: string;           // full combo text e.g. "원투 슥 투훅투"
  chunks: Chunk[];
}

export interface Timestamp {
  start_ms: number;
  end_ms: number;
  text: string;
}

export interface Round {
  round: number;
  segments: Segment[];
  audio_url?: string;
  timestamps?: Timestamp[];
}

export interface PlanDetail {
  rounds: Round[];
}

export interface PlanResponse {
  id: string;
  template_name: string;
  template_topic: string;
  rounds: number;
  round_duration_sec: number;
  rest_sec: number;
  plan: PlanDetail;
  audio_ready?: boolean;
  day_number?: number;
  theme?: string;
  coach_comment?: string;
}

export interface ProgramDaySummary {
  id: string;
  day_number: number;
  theme: string;
  theme_description: string;
  coach_comment: string;
}

export interface TodayData {
  streak: number;
  day_number: number;
  day_total: number;
  theme: string;
  theme_description: string;
  coach_comment: string;
  level: string;
  week: number;
  next_day_preview: { day_number: number; theme: string; is_cycle_restart?: boolean } | null;
  program_days: ProgramDaySummary[];
}

export interface SessionSummary {
  id: string;
  drill_plan_id: string;
  started_at: string;
  completed_at: string | null;
  total_duration_sec: number;
  rounds_completed: number;
  rounds_total: number;
  segments_delivered: number;
  status: string;
}

export interface SessionLogRequest {
  drill_plan_id: string;
  started_at: string;
  completed_at: string;
  total_duration_sec: number;
  rounds_completed: number;
  rounds_total: number;
  segments_delivered: number;
  status: string;
}

export interface SessionLogResponse {
  id: string;
  started_at: string;
  completed_at: string | null;
  total_duration_sec: number;
  rounds_completed: number;
  rounds_total: number;
  segments_delivered: number;
  status: string;
}

export const fetchToday = (): Promise<TodayData> =>
  apiFetch<TodayData>('/api/today');

export const generatePlan = (body: PlanRequest): Promise<PlanResponse> =>
  apiFetch<PlanResponse>('/api/sessions/plan', {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const logSession = (body: SessionLogRequest): Promise<SessionLogResponse> =>
  apiFetch<SessionLogResponse>('/api/sessions/log', {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const fetchSessions = (limit = 20, offset = 0): Promise<SessionSummary[]> =>
  apiFetch<SessionSummary[]>(`/api/sessions?limit=${limit}&offset=${offset}`);
