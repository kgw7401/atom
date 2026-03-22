import React, { useEffect, useRef, useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import * as Speech from 'expo-speech';
import * as Haptics from 'expo-haptics';
import { Audio } from 'expo-av';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSequence,
  withTiming,
  FadeInDown,
  FadeOut,
} from 'react-native-reanimated';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { PlanResponse, Round, Segment } from '../api/session';
import { useSettingsStore } from '../store/settingsStore';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';

type Props = NativeStackScreenProps<any, 'ActiveSession'>;

// ── Phase definitions (fixed 10-min session) ─────────────────────────
interface Phase {
  key: string;
  label: string;
  labelKo: string;
  duration: number;  // seconds
  bgColor: string;
  roundIndex?: number;  // which plan round to play (0-3)
}

const PHASES: Phase[] = [
  { key: 'intro',    label: 'INTRO',               labelKo: '준비',   duration: 40,  bgColor: COLORS.BG },
  { key: 'round1',   label: 'ROUND 1',             labelKo: '적응',   duration: 120, bgColor: COLORS.PHASE_ROUND, roundIndex: 0 },
  { key: 'rest1',    label: 'REST',                labelKo: '휴식',   duration: 30,  bgColor: COLORS.PHASE_REST },
  { key: 'round2',   label: 'ROUND 2',             labelKo: '적용',   duration: 120, bgColor: COLORS.PHASE_ROUND, roundIndex: 1 },
  { key: 'rest2',    label: 'REST',                labelKo: '휴식',   duration: 30,  bgColor: COLORS.PHASE_REST },
  { key: 'round3',   label: 'ROUND 3',             labelKo: '몰입',   duration: 120, bgColor: COLORS.PHASE_ROUND, roundIndex: 2 },
  { key: 'finisher', label: 'FINISHER',            labelKo: '폭발',   duration: 90,  bgColor: COLORS.PHASE_FINISHER, roundIndex: 3 },
  { key: 'outro',    label: 'COOL DOWN',           labelKo: '마무리', duration: 50,  bgColor: COLORS.BG },
];

const TOTAL_SESSION_SEC = PHASES.reduce((sum, p) => sum + p.duration, 0); // 600s = 10min

export default function ActiveSessionScreen({ route, navigation }: Props) {
  const { plan: planResponse, today } = route.params as {
    plan: PlanResponse;
    today?: any;
  };

  const serverUrl = useSettingsStore((s) => s.serverUrl);
  const planRounds = planResponse.plan.rounds; // R1, R2, R3, Finisher

  const [countdown, setCountdown] = useState(3);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [phaseSecondsLeft, setPhaseSecondsLeft] = useState(PHASES[0].duration);
  const [totalSecondsLeft, setTotalSecondsLeft] = useState(TOTAL_SESSION_SEC);
  const [currentText, setCurrentText] = useState('');
  const [segmentKey, setSegmentKey] = useState(0);
  const [paused, setPaused] = useState(false);

  const abortRef = useRef(false);
  const pausedRef = useRef(false);
  const segmentsDeliveredRef = useRef(0);
  const startedAtRef = useRef(new Date());
  const soundRef = useRef<Audio.Sound | null>(null);

  const timerScale = useSharedValue(1);
  const timerAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: timerScale.value }],
  }));

  const currentPhase = PHASES[phaseIndex] || PHASES[0];

  // ── Utility: sleep respecting pause/abort ──────────────────────────
  const sleepMs = useCallback((ms: number): Promise<void> =>
    new Promise<void>((resolve) => {
      const target = Date.now() + ms;
      const tick = () => {
        if (abortRef.current) { resolve(); return; }
        if (pausedRef.current) { setTimeout(tick, 100); return; }
        if (Date.now() >= target) resolve();
        else setTimeout(tick, Math.min(50, target - Date.now()));
      };
      tick();
    }), []);

  // ── Stop current sound ─────────────────────────────────────────────
  const stopSound = useCallback(async () => {
    if (soundRef.current) {
      try { await soundRef.current.stopAsync(); } catch {}
      try { await soundRef.current.unloadAsync(); } catch {}
      soundRef.current = null;
    }
  }, []);

  // ── Pause toggle ───────────────────────────────────────────────────
  const handlePauseToggle = useCallback(async () => {
    const next = !pausedRef.current;
    pausedRef.current = next;
    setPaused(next);
    if (next) {
      Speech.stop();
      if (soundRef.current) {
        try { await soundRef.current.pauseAsync(); } catch {}
      }
    } else {
      if (soundRef.current) {
        try { await soundRef.current.playAsync(); } catch {}
      }
    }
  }, []);

  // ── Abort ──────────────────────────────────────────────────────────
  const handleAbort = () => {
    Alert.alert('세션 중단', '정말 중단하시겠습니까?', [
      { text: '계속', style: 'cancel' },
      { text: '중단', style: 'destructive', onPress: async () => {
        abortRef.current = true;
        Speech.stop();
        await stopSound();
      }},
    ]);
  };

  // ── Navigation guard ───────────────────────────────────────────────
  useEffect(() => {
    const unsubscribe = navigation.addListener('beforeRemove', (e: any) => {
      if (abortRef.current) return;
      e.preventDefault();
      handleAbort();
    });
    return unsubscribe;
  }, [navigation, stopSound]);

  // ── Timer pulse animation ──────────────────────────────────────────
  useEffect(() => {
    const interval = setInterval(() => {
      timerScale.value = withSequence(
        withTiming(1.015, { duration: 150 }),
        withTiming(1, { duration: 150 }),
      );
    }, 1000);
    return () => clearInterval(interval);
  }, [timerScale]);

  // ── Speak helper ───────────────────────────────────────────────────
  const speakAsync = (text: string): Promise<void> =>
    new Promise<void>((resolve) => {
      Speech.speak(text, {
        language: 'ko-KR',
        rate: 1.1,
        pitch: 1.0,
        onDone: resolve,
        onStopped: resolve,
        onError: () => resolve(),
      });
    });

  // ── Play assembled round audio with timestamp sync ─────────────────
  const playRoundAudio = async (round: Round): Promise<void> => {
    const audioUrl = round.audio_url!;
    const timestamps = round.timestamps!;
    const fullUrl = serverUrl + audioUrl;

    try {
      const { sound } = await Audio.Sound.createAsync({ uri: fullUrl });
      soundRef.current = sound;

      let lastSegIdx = -1;
      let finished = false;

      sound.setOnPlaybackStatusUpdate((status) => {
        if (!status.isLoaded || finished) return;
        if (status.didJustFinish) { finished = true; return; }

        const pos = status.positionMillis;
        for (let j = timestamps.length - 1; j >= 0; j--) {
          if (pos >= timestamps[j].start_ms) {
            if (j !== lastSegIdx) {
              lastSegIdx = j;
              segmentsDeliveredRef.current++;
              setCurrentText(timestamps[j].text);
              setSegmentKey((k) => k + 1);
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            }
            break;
          }
        }
      });

      await sound.playAsync();

      await new Promise<void>((resolve) => {
        const interval = setInterval(async () => {
          if (abortRef.current || finished) {
            clearInterval(interval);
            resolve();
            return;
          }
          try {
            const s = await sound.getStatusAsync();
            if (!s.isLoaded || s.didJustFinish) {
              clearInterval(interval);
              resolve();
            }
          } catch { clearInterval(interval); resolve(); }
        }, 250);
      });

      await sound.unloadAsync();
      soundRef.current = null;
    } catch {
      soundRef.current = null;
    }
  };

  // ── Play single audio chunk ────────────────────────────────────────
  const playChunk = async (clipUrl: string): Promise<void> => {
    const fullUrl = serverUrl + clipUrl;
    try {
      const { sound } = await Audio.Sound.createAsync({ uri: fullUrl });
      soundRef.current = sound;
      await sound.playAsync();
      await new Promise<void>((resolve) => {
        sound.setOnPlaybackStatusUpdate((status) => {
          if (!status.isLoaded || status.didJustFinish) resolve();
        });
      });
      await sound.unloadAsync();
      soundRef.current = null;
    } catch {
      soundRef.current = null;
    }
  };

  // ── Play segment chunk-by-chunk ────────────────────────────────────
  const playSegmentFallback = async (segment: Segment): Promise<void> => {
    if (abortRef.current) return;

    segmentsDeliveredRef.current++;
    setCurrentText(segment.text);
    setSegmentKey((k) => k + 1);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    const hasAudio = segment.chunks.some((c) => c.clip_url);

    if (hasAudio) {
      for (let i = 0; i < segment.chunks.length; i++) {
        if (abortRef.current) break;
        const chunk = segment.chunks[i];
        if (chunk.clip_url) {
          await playChunk(chunk.clip_url);
        } else {
          await sleepMs(chunk.duration_ms || 300);
        }
        if (i < segment.chunks.length - 1) await sleepMs(150);
      }
      await sleepMs(300);
    } else {
      await speakAsync(segment.text);
      await sleepMs(300);
    }
  };

  // ── Play a round (assembled or fallback) ───────────────────────────
  const playRound = async (round: Round): Promise<void> => {
    if (round.audio_url && round.timestamps?.length) {
      await playRoundAudio(round);
    } else {
      for (const segment of round.segments) {
        if (abortRef.current) break;
        await playSegmentFallback(segment);
      }
    }
  };

  // ── Countdown then start ───────────────────────────────────────────
  useEffect(() => {
    if (sessionStarted) return;
    if (countdown <= 0) {
      setSessionStarted(true);
      return;
    }
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown, sessionStarted]);

  // ── Main session loop ──────────────────────────────────────────────
  useEffect(() => {
    if (!sessionStarted) return;

    let phaseTimer: ReturnType<typeof setInterval> | null = null;
    let totalTimer: ReturnType<typeof setInterval> | null = null;

    const runSession = async () => {
      await Audio.setAudioModeAsync({ playsInSilentModeIOS: true });
      startedAtRef.current = new Date();

      let totalLeft = TOTAL_SESSION_SEC;
      totalTimer = setInterval(() => {
        if (pausedRef.current) return;
        totalLeft = Math.max(0, totalLeft - 1);
        setTotalSecondsLeft(totalLeft);
      }, 1000);

      for (let pi = 0; pi < PHASES.length; pi++) {
        if (abortRef.current) break;

        const phase = PHASES[pi];
        setPhaseIndex(pi);
        setCurrentText('');

        let phaseSecs = phase.duration;
        setPhaseSecondsLeft(phaseSecs);

        if (phaseTimer) clearInterval(phaseTimer);
        phaseTimer = setInterval(() => {
          if (pausedRef.current) return;
          phaseSecs = Math.max(0, phaseSecs - 1);
          setPhaseSecondsLeft(phaseSecs);
        }, 1000);

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

        if (phase.key === 'intro') {
          // Intro phase: coach intro
          setCurrentText('준비');
          await speakAsync('시작합니다');
          await sleepMs((phase.duration - 5) * 1000);

        } else if (phase.key.startsWith('rest')) {
          // Rest phase
          setCurrentText('쉬어');
          await speakAsync('쉬어');
          await sleepMs((phase.duration - 3) * 1000);
          if (!abortRef.current) {
            setCurrentText('다음 라운드 준비');
            await speakAsync('다음 라운드 준비');
            await sleepMs(2000);
          }

        } else if (phase.key === 'outro') {
          // Outro phase
          setCurrentText('수고했어!');
          await speakAsync('수고했어! 오늘도 잘했다.');
          await sleepMs((phase.duration - 5) * 1000);

        } else if (phase.roundIndex !== undefined) {
          // Round or Finisher: play segments from plan
          const round = planRounds[phase.roundIndex];
          if (round) {
            await playRound(round);
          }
          // Fill remaining phase time with silence
          // (audio may finish before phase duration)
          while (phaseSecs > 1 && !abortRef.current) {
            await sleepMs(500);
          }
        }

        if (phaseTimer) clearInterval(phaseTimer);
      }

      // Session complete
      if (totalTimer) clearInterval(totalTimer);
      if (phaseTimer) clearInterval(phaseTimer);
      Speech.stop();
      await stopSound();

      const completedAt = new Date();
      const durationSec = (completedAt.getTime() - startedAtRef.current.getTime()) / 1000;
      const status = abortRef.current ? 'abandoned' : 'completed';

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
      abortRef.current = true;

      navigation.replace('SessionEnd', {
        status,
        rounds: 3,
        roundsTotal: 3,
        segments: segmentsDeliveredRef.current,
        duration: Math.round(durationSec),
        today,
        planResponse,
        logPayload: {
          drill_plan_id: planResponse.id,
          started_at: startedAtRef.current.toISOString(),
          completed_at: completedAt.toISOString(),
          total_duration_sec: durationSec,
          rounds_completed: 3,
          rounds_total: 3,
          segments_delivered: segmentsDeliveredRef.current,
          status,
        },
      });
    };

    runSession();
    return () => {
      if (phaseTimer) clearInterval(phaseTimer);
      if (totalTimer) clearInterval(totalTimer);
      Speech.stop();
      stopSound();
    };
  }, [sessionStarted]);

  // ── Countdown screen ───────────────────────────────────────────────
  if (!sessionStarted) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={styles.countdownNumber}>{countdown}</Text>
        <Text style={styles.countdownLabel}>준비</Text>
      </View>
    );
  }

  const pMM = String(Math.floor(phaseSecondsLeft / 60)).padStart(2, '0');
  const pSS = String(phaseSecondsLeft % 60).padStart(2, '0');
  const tMM = String(Math.floor(totalSecondsLeft / 60)).padStart(2, '0');
  const tSS = String(totalSecondsLeft % 60).padStart(2, '0');

  const isRound = currentPhase.key.startsWith('round') || currentPhase.key === 'finisher';

  // Which rounds are completed
  const completedRounds = PHASES.slice(0, phaseIndex + 1)
    .filter(p => p.key.startsWith('round') || p.key === 'finisher').length;
  const roundPhases = PHASES.filter(p => p.key.startsWith('round') || p.key === 'finisher');

  return (
    <View style={[styles.container, { backgroundColor: currentPhase.bgColor }]}>
      {/* Phase label */}
      <View style={styles.header}>
        <Text style={styles.phaseLabel}>
          {currentPhase.label}{currentPhase.labelKo ? ` · ${currentPhase.labelKo}` : ''}
        </Text>
        {paused && <Text style={styles.pausedBadge}>PAUSED</Text>}
      </View>

      {/* Total session time */}
      <Text style={styles.totalTime}>{tMM}:{tSS}</Text>

      {/* Phase timer */}
      <View style={styles.timerSection}>
        <Animated.View style={timerAnimatedStyle}>
          <Text style={[styles.timer, !isRound && styles.timerDim]}>
            {pMM}:{pSS}
          </Text>
        </Animated.View>
      </View>

      <View style={styles.divider} />

      {/* Current instruction */}
      <View style={styles.instrContainer}>
        {currentText ? (
          <Animated.View
            key={segmentKey}
            entering={FadeInDown.duration(120)}
            exiting={FadeOut.duration(80)}
          >
            <Text style={styles.segmentText}>{currentText}</Text>
          </Animated.View>
        ) : (
          <Text style={styles.waiting}>
            {currentPhase.key === 'intro' ? '준비...' :
             currentPhase.key.startsWith('rest') ? '잠시 쉬세요' :
             currentPhase.key === 'outro' ? '마무리' : ''}
          </Text>
        )}
      </View>

      {/* Round progress dots */}
      <View style={styles.dots}>
        {roundPhases.map((rp, i) => {
          const rpIdx = PHASES.indexOf(rp);
          const isDone = rpIdx < phaseIndex;
          const isCurrent = rpIdx === phaseIndex;
          return (
            <View key={rp.key} style={styles.dotWrapper}>
              <View
                style={[
                  styles.dot,
                  isDone && styles.dotDone,
                  isCurrent && styles.dotActive,
                  !isDone && !isCurrent && styles.dotPending,
                  rp.key === 'finisher' && styles.dotFinisher,
                ]}
              />
              <Text style={styles.dotLabel}>
                {rp.key === 'finisher' ? 'F' : `R${i + 1}`}
              </Text>
            </View>
          );
        })}
      </View>

      {/* Controls */}
      <View style={styles.controlRow}>
        <TouchableOpacity style={styles.controlBtn} onPress={handlePauseToggle} activeOpacity={0.7}>
          <Text style={styles.controlBtnText}>{paused ? '재개' : '일시정지'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.controlBtn} onPress={handleAbort} activeOpacity={0.7}>
          <Text style={[styles.controlBtnText, styles.abortText]}>중단</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: SPACING.PADDING_SCREEN, justifyContent: 'space-between' },

  // Countdown
  countdownNumber: {
    color: COLORS.RED,
    fontSize: 120,
    fontWeight: '200',
    letterSpacing: 8,
  },
  countdownLabel: {
    color: COLORS.TEXT_2,
    ...TYPOGRAPHY.TITLE,
    marginTop: 16,
  },

  // Header
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 40 },
  phaseLabel: {
    color: COLORS.TEXT_2,
    ...TYPOGRAPHY.SECTION_LABEL,
    letterSpacing: 2,
  },
  pausedBadge: { color: COLORS.GOLD, fontSize: 12, fontWeight: '600' },

  // Total time
  totalTime: {
    color: COLORS.TEXT_3,
    ...TYPOGRAPHY.META,
    textAlign: 'center',
    marginTop: 8,
  },

  // Timer
  timerSection: { alignItems: 'center', justifyContent: 'center', height: 140 },
  timer: {
    color: COLORS.TEXT_1,
    fontSize: TYPOGRAPHY.TIMER.fontSize,
    fontWeight: TYPOGRAPHY.TIMER.fontWeight,
    letterSpacing: TYPOGRAPHY.TIMER.letterSpacing,
  },
  timerDim: { color: '#3a4a6a' },

  divider: { height: 1, backgroundColor: '#ffffff12' },

  // Instructions
  instrContainer: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  segmentText: {
    fontSize: TYPOGRAPHY.COMBO.fontSize,
    fontWeight: TYPOGRAPHY.COMBO.fontWeight,
    textAlign: 'center',
    letterSpacing: 2,
    color: COLORS.TEXT_1,
  },
  waiting: { color: COLORS.TEXT_GHOST, fontSize: 24, textAlign: 'center' },

  // Dots
  dots: { flexDirection: 'row', justifyContent: 'center', gap: 16, marginBottom: 16 },
  dotWrapper: { alignItems: 'center', gap: 4 },
  dot: { width: 12, height: 12, borderRadius: 6 },
  dotDone: { backgroundColor: COLORS.RED },
  dotActive: { backgroundColor: COLORS.ORANGE },
  dotPending: { backgroundColor: COLORS.TEXT_GHOST },
  dotFinisher: { borderRadius: 3 },
  dotLabel: { color: COLORS.TEXT_3, ...TYPOGRAPHY.MICRO },

  // Controls
  controlRow: { flexDirection: 'row', gap: 12, marginBottom: 32 },
  controlBtn: {
    flex: 1,
    paddingVertical: 20,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff0d',
    borderWidth: 1,
    borderColor: '#ffffff18',
  },
  controlBtnText: { color: COLORS.TEXT_1, fontSize: 15, fontWeight: '600' },
  abortText: { color: COLORS.TEXT_GHOST },
});
