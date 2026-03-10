/**
 * Base API client — wraps fetch with server URL from settings store.
 */
import { useSettingsStore } from '../store/settingsStore';

export function getBaseUrl(): string {
  return useSettingsStore.getState().serverUrl;
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}
