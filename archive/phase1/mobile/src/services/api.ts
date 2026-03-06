import axios from 'axios';
import type {
  ScriptResponse,
  SessionResponse,
  SessionStatus,
  SessionReport,
  UserResponse,
  TwinResponse,
} from '../types';

// Change this to your local machine IP for testing on device
// Using network IP works for both Simulator and Physical Device
const API_BASE_URL = 'http://192.168.0.4:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const ApiClient = {
  // User
  createUser: async (deviceId: string): Promise<UserResponse> => {
    const { data } = await api.post('/users', { device_id: deviceId });
    return data;
  },

  getTwin: async (userId: string): Promise<TwinResponse> => {
    const { data } = await api.get(`/users/${userId}/twin`);
    return data;
  },

  // Script
  generateScript: async (level: number, duration: number): Promise<ScriptResponse> => {
    const { data } = await api.post('/scripts/generate', {
      level,
      duration_seconds: duration,
    });
    return data;
  },

  // Session
  createSession: async (
    userId: string,
    scriptId: string,
    startedAt: string
  ): Promise<SessionResponse> => {
    const { data } = await api.post('/sessions', {
      user_id: userId,
      script_id: scriptId,
      started_at: startedAt,
    });
    return data;
  },

  uploadVideo: async (
    sessionId: string,
    videoUri: string,
    onProgress?: (progress: number) => void
  ): Promise<SessionStatus> => {
    const formData = new FormData();

    // @ts-ignore - React Native FormData handles file objects differently
    formData.append('video', {
      uri: videoUri,
      type: 'video/mp4',
      name: 'video.mp4',
    });

    const { data } = await api.post(`/sessions/${sessionId}/upload-video`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });

    return data;
  },

  getSessionStatus: async (sessionId: string): Promise<SessionStatus> => {
    const { data } = await api.get(`/sessions/${sessionId}/status`);
    return data;
  },

  getSessionReport: async (sessionId: string): Promise<SessionReport> => {
    const { data } = await api.get(`/sessions/${sessionId}/report`);
    return data;
  },
};
