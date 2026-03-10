import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';

const STORAGE_KEY = 'atom_server_url';
const DEFAULT_URL = 'http://localhost:8000';

interface SettingsState {
  serverUrl: string;
  setServerUrl: (url: string) => Promise<void>;
  loadServerUrl: () => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  serverUrl: DEFAULT_URL,

  setServerUrl: async (url: string) => {
    const clean = url.replace(/\/$/, ''); // strip trailing slash
    await AsyncStorage.setItem(STORAGE_KEY, clean);
    set({ serverUrl: clean });
  },

  loadServerUrl: async () => {
    const stored = await AsyncStorage.getItem(STORAGE_KEY);
    if (stored) set({ serverUrl: stored });
  },
}));
