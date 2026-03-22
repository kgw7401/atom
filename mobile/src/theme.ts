export const COLORS = {
  BG: '#0A0A0A',
  SURFACE: '#151515',
  RED: '#E5383B',
  RED_BG: '#1A0608',
  GREEN: '#4caf50',
  ORANGE: '#FF6B35',
  GOLD: '#FFD166',
  GOLD_BG: '#2a2a1a',
  BLUE: '#4EA8DE',
  TEXT_1: '#f0f0f0',
  TEXT_2: '#999',
  TEXT_3: '#555',
  TEXT_GHOST: '#333',
  BORDER: '#242424',
  // Phase background tints
  PHASE_ROUND: '#0A140A',
  PHASE_REST: '#080D18',
  PHASE_FINISHER: '#1A0A00',
} as const;

export const SPACING = {
  PADDING_SCREEN: 24,
  PADDING_CARD: 16,
  GAP_SECTION: 24,
  GAP_ITEM: 8,
  RADIUS_CARD: 12,
  RADIUS_BADGE: 20,
} as const;

export const TYPOGRAPHY = {
  TIMER: { fontSize: 96, fontWeight: '200' as const, letterSpacing: 8 },
  COMBO: { fontSize: 52, fontWeight: '700' as const },
  STREAK_NUMBER: { fontSize: 42, fontWeight: '900' as const },
  APP_TITLE: { fontSize: 32, fontWeight: '900' as const, letterSpacing: 6 },
  TITLE: { fontSize: 28, fontWeight: '700' as const },
  PLAN_FOCUS: { fontSize: 22, fontWeight: '700' as const },
  COACH_BODY: { fontSize: 18, lineHeight: 28 },
  CARD_TITLE: { fontSize: 17, fontWeight: '600' as const },
  BODY: { fontSize: 16 },
  STAT_VALUE: { fontSize: 22, fontWeight: '700' as const },
  META: { fontSize: 13 },
  SECTION_LABEL: { fontSize: 11, fontWeight: '700' as const, letterSpacing: 1.5 },
  MICRO: { fontSize: 10 },
} as const;
