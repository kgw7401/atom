import type { ScriptResponse } from '../types';

export type RootStackParamList = {
  Home: undefined;
  SessionSetup: { userId: string };
  DrillSession: { sessionId: string; script: ScriptResponse };
  Uploading: { sessionId: string; videoUri: string };
  Report: { sessionId: string };
  Twin: { userId: string };
};
