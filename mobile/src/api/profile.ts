import { apiFetch } from './client';

export interface Profile {
  id: string;
  experience_level: string;
  goal: string;
  total_sessions: number;
  total_training_minutes: number;
  last_session_at: string | null;
  combo_exposure_json: Record<string, number>;
  template_preference_json: Record<string, number>;
  session_frequency: number;
}

export interface ProfileUpdate {
  experience_level?: string;
  goal?: string;
}

export const fetchProfile = (): Promise<Profile> =>
  apiFetch<Profile>('/api/profile');

export const updateProfile = (body: ProfileUpdate): Promise<Profile> =>
  apiFetch<Profile>('/api/profile', { method: 'PUT', body: JSON.stringify(body) });
