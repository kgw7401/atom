// API types matching server schemas

export interface Instruction {
  t: number;
  type: string;
  action: string;
  audio_key: string;
  display: string;
}

export interface ScriptResponse {
  script_id: string;
  instructions: Instruction[];
  level: number;
  total_instructions: number;
  estimated_duration_seconds: number;
}

export interface SessionResponse {
  session_id: string;
  upload_url: string;
  status: string;
}

export interface SessionStatus {
  status: 'created' | 'analyzing' | 'completed' | 'failed';
  progress: number;
}

export interface InstructionResult {
  index: number;
  t: number;
  type: string;
  command: string;
  status: 'success' | 'partial' | 'missed';
  score: number;
  reaction_time: number | null;
  detected_actions: string[];
  feedback: string;
}

export interface ReportSummary {
  total_instructions: number;
  completed: number;
  success: number;
  partial: number;
  missed: number;
  attack_accuracy: number;
  defense_accuracy: number;
  avg_reaction_time: number;
}

export interface CoachingFeedback {
  strengths: string[];
  weaknesses: string[];
  next_session: string;
}

export interface SessionReport {
  session_id: string;
  overall_score: number;
  summary: ReportSummary;
  instructions: InstructionResult[];
  coaching: CoachingFeedback;
}

export interface UserResponse {
  user_id: string;
}

export interface ActionStat {
  accuracy: number;
  avg_reaction: number;
  trend: string;
  total_attempts: number;
}

export interface Weakness {
  action: string;
  metric: string;
  value: number;
  threshold: number;
  severity: string;
}

export interface TwinResponse {
  total_sessions: number;
  per_action_stats: Record<string, ActionStat>;
  weaknesses: Weakness[];
  growth_curves: {
    scores: number[];
  };
}
