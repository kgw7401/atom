import { apiFetch } from './client';

export interface Template {
  id: string;
  name: string;
  display_name: string;
  description: string;
  default_rounds: number;
  default_round_duration_sec: number;
  default_rest_sec: number;
  combo_complexity_range: [number, number];
  combo_include_defense: boolean;
  pace_interval_sec: [number, number];
}

export const fetchTemplates = (): Promise<Template[]> =>
  apiFetch<Template[]>('/api/templates');
