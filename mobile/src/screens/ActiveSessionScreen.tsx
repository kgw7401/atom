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
  withRepeat,
  FadeInDown,
  FadeOut,
} from 'react-native-reanimated';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { PlanDetail, Segment } from '../api/session';
import { useSettingsStore } from '../store/settingsStore';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';

type Props = NativeStackScreenProps<any, 'ActiveSession'>;

const BG_ROUND  = '#0a140a';
const BG_URGENT = '#160808';
const BG_REST   = '#080d18';

const CHUNK_GAP_MS = 150;   // gap between chunks within a combo
const SEGMENT_PAUSE_MS = 300; // pause after last chunk before next segment

export default function ActiveSessionScreen({ route, navigation }: Props) {
  const { planId, plan, round_duration_sec, rest_sec } = route.params as {
    planId: string;
    plan: PlanDetail;
    round_duration_sec: number;
    rest_sec: number;
  };

  const serverUrl = useSettingsStore((s) => s.serverUrl);
  const rounds = plan.rounds;

  const [phase, setPhase] = useState<'round' | 'rest'>('round');
  const [roundNum, setRoundNum] = useState(rounds[0]?.round ?? 1);
  const [secondsLeft, setSecondsLeft] = useState(round_duration_sec);
  const [currentText, setCurrentText] = useState('');
  const [segmentKey, setSegmentKey] = useState(0);
  const [paused, setPaused] = useState(false);

  const abortRef = useRef(false);
  const pausedRef = useRef(false);
  const roundsCompletedRef = useRef(0);
  const segmentsDeliveredRef = useRef(0);
  const startedAtRef = useRef(new Date());
  const soundRef = useRef<Audio.Sound | null>(null);

  const timerScale = useSharedValue(1);
  const urgentOpacity = useSharedValue(0);

  const timerAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: timerScale.value }],
  }));
  const urgentOverlayStyle = useAnimatedStyle(() => ({ opacity: urgentOpacity.value }));

  const bgColor =
    phase === 'rest' ? BG_REST
    : phase === 'round' && secondsLeft <= 30 && secondsLeft > 0 ? BG_URGENT
    : BG_ROUND;

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
    const unsubscribe = navigation.addListener('beforeRemove', (e) => {
      if (abortRef.current) return;
      e.preventDefault();
      Alert.alert('세션 중단', '정말 중단하시겠습니까?', [
        { text: '계속', style: 'cancel' },
        { text: '중단', style: 'destructive', onPress: async () => {
          abortRef.current = true;
          Speech.stop();
          await stopSound();
        }},
      ]);
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

  // ── Urgent phase overlay ───────────────────────────────────────────
  useEffect(() => {
    if (phase === 'round' && secondsLeft <= 30 && secondsLeft > 0) {
      urgentOpacity.value = withRepeat(
        withSequence(withTiming(0.15, { duration: 500 }), withTiming(0, { duration: 500 })),
        -1, false,
      );
    } else {
      urgentOpacity.value = withTiming(0, { duration: 300 });
    }
  }, [phase, secondsLeft <= 30]);

  // ── Main session loop ──────────────────────────────────────────────
  useEffect(() => {
    let countdownTimer: ReturnType<typeof setInterval> | null = null;

    // ── Play a single audio chunk ─────────────────────────────────────
    const playChunk = async (clipUrl: string): Promise<void> => {
      const fullUrl = serverUrl + clipUrl;
      try {
        const { sound } = await Audio.Sound.createAsync({ uri: fullUrl });
        soundRef.current = sound;

        await sound.playAsync();

        // Wait for playback to finish
        await new Promise<void>((resolve) => {
          sound.setOnPlaybackStatusUpdate((status) => {
            if (!status.isLoaded || status.didJustFinish) {
              resolve();
            }
          });
        });

        await sound.unloadAsync();
        soundRef.current = null;
      } catch {
        // Audio failed — skip this chunk
        soundRef.current = null;
      }
    };

    // ── Speak with expo-speech (fallback when no audio) ───────────────
    const speakAsync = (text: string): Promise<void> =>
      new Promise<void>((resolve) => {
        Speech.speak(text, {
          language: 'ko-KR',
          rate: 1.1,
          pitch: 1.0,
          onDone: resolve,
          onStopped: resolve,
          onError: resolve,
        });
      });

    // ── Play a segment (chunk-by-chunk or TTS fallback) ───────────────
    const playSegment = async (segment: Segment): Promise<void> => {
      if (abortRef.current) return;

      segmentsDeliveredRef.current++;
      setCurrentText(segment.text);
      setSegmentKey((k) => k + 1);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      const hasAudio = segment.chunks.some((c) => c.clip_url);

      if (hasAudio) {
        // Play chunks with gaps
        for (let i = 0; i < segment.chunks.length; i++) {
          if (abortRef.current) break;
          const chunk = segment.chunks[i];

          if (chunk.clip_url) {
            await playChunk(chunk.clip_url);
          } else {
            // Missing chunk — use duration as silence
            await sleepMs(chunk.duration_ms || 300);
          }

          // Gap between chunks (not after last)
          if (i < segment.chunks.length - 1) {
            await sleepMs(CHUNK_GAP_MS);
          }
        }

        // Pause after segment = total chunk duration + 300ms
        await sleepMs(SEGMENT_PAUSE_MS);
      } else {
        // No audio available — fall back to TTS
        await speakAsync(segment.text);
        await sleepMs(SEGMENT_PAUSE_MS);
      }
    };

    // ── Session runner ────────────────────────────────────────────────
    const runSession = async () => {
      await Audio.setAudioModeAsync({ playsInSilentModeIOS: true });

      for (let i = 0; i < rounds.length; i++) {
        if (abortRef.current) break;

        const round = rounds[i];
        let secsLeft = round_duration_sec;

        setRoundNum(round.round);
        setPhase('round');
        setCurrentText('');
        setSecondsLeft(secsLeft);

        // Start countdown timer
        if (countdownTimer) clearInterval(countdownTimer);
        countdownTimer = setInterval(() => {
          if (pausedRef.current) return;
          secsLeft = Math.max(0, secsLeft - 1);
          setSecondsLeft(secsLeft);
        }, 1000);

        // Play segments sequentially
        for (const segment of round.segments) {
          if (abortRef.current) break;
          await playSegment(segment);
        }

        if (countdownTimer) clearInterval(countdownTimer);
        roundsCompletedRef.current++;
        setCurrentText('');

        if (abortRef.current) break;

        // Rest between rounds
        const isLastRound = i === rounds.length - 1;
        if (!isLastRound && rest_sec > 0) {
          let restSecs = rest_sec;
          setPhase('rest');
          setSecondsLeft(restSecs);
          await speakAsync('휴식');
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

          countdownTimer = setInterval(() => {
            if (pausedRef.current) return;
            restSecs = Math.max(0, restSecs - 1);
            setSecondsLeft(restSecs);
          }, 1000);

          await sleepMs(rest_sec * 1000);
          if (countdownTimer) clearInterval(countdownTimer);
        }
      }

      // Session complete
      if (countdownTimer) clearInterval(countdownTimer);
      Speech.stop();
      await stopSound();

      const completedAt = new Date();
      const durationSec = (completedAt.getTime() - startedAtRef.current.getTime()) / 1000;
      const status = abortRef.current ? 'abandoned' : 'completed';

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
      abortRef.current = true;

      navigation.replace('SessionEnd', {
        status,
        rounds: roundsCompletedRef.current,
        roundsTotal: rounds.length,
        segments: segmentsDeliveredRef.current,
        duration: Math.round(durationSec),
        logPayload: {
          drill_plan_id: planId,
          started_at: startedAtRef.current.toISOString(),
          completed_at: completedAt.toISOString(),
          total_duration_sec: durationSec,
          rounds_completed: roundsCompletedRef.current,
          rounds_total: rounds.length,
          segments_delivered: segmentsDeliveredRef.current,
          status,
        },
      });
    };

    runSession();
    return () => {
      if (countdownTimer) clearInterval(countdownTimer);
      Speech.stop();
      stopSound();
    };
  }, []);

  const mm = String(Math.floor(secondsLeft / 60)).padStart(2, '0');
  const ss = String(secondsLeft % 60).padStart(2, '0');
  const isRest = phase === 'rest';

  return (
    <View style={[styles.container, { backgroundColor: bgColor }]}>
      <Animated.View style={[styles.urgentOverlay, urgentOverlayStyle]} pointerEvents="none" />

      <View style={styles.header}>
        <Text style={styles.roundLabel}>
          {isRest ? 'REST' : `ROUND ${roundNum} / ${rounds.length}`}
        </Text>
        {paused && <Text style={styles.pausedBadge}>PAUSED</Text>}
      </View>

      <View style={styles.timerSection}>
        <Animated.View style={timerAnimatedStyle}>
          <Text style={[styles.timer, isRest && styles.timerRest]}>
            {mm}:{ss}
          </Text>
        </Animated.View>
      </View>

      <View style={styles.divider} />

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
            {isRest ? '잠시 쉬세요.' : '준비...'}
          </Text>
        )}
      </View>

      <View style={styles.dots}>
        {rounds.map((_, i) => (
          <View
            key={i}
            style={[
              styles.dot,
              i < roundsCompletedRef.current ? styles.dotDone
              : i === roundNum - 1 ? styles.dotActive
              : styles.dotPending,
            ]}
          />
        ))}
      </View>

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
  urgentOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: '#ff0000' },

  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 40 },
  roundLabel: {
    color: COLORS.TEXT_GHOST,
    fontSize: TYPOGRAPHY.SECTION_LABEL.fontSize,
    fontWeight: TYPOGRAPHY.SECTION_LABEL.fontWeight,
    textTransform: 'uppercase',
    letterSpacing: 2,
  },
  pausedBadge: { color: COLORS.GOLD, fontSize: 12, fontWeight: '600' },

  timerSection: { alignItems: 'center', justifyContent: 'center', height: 140 },
  timer: {
    color: COLORS.TEXT_1,
    fontSize: TYPOGRAPHY.TIMER.fontSize,
    fontWeight: TYPOGRAPHY.TIMER.fontWeight,
    letterSpacing: TYPOGRAPHY.TIMER.letterSpacing,
  },
  timerRest: { color: '#3a4a6a' },

  divider: { height: 1, backgroundColor: '#ffffff12' },

  instrContainer: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  segmentText: {
    fontSize: TYPOGRAPHY.COMBO.fontSize,
    fontWeight: TYPOGRAPHY.COMBO.fontWeight,
    textAlign: 'center',
    letterSpacing: 2,
    color: COLORS.TEXT_1,
  },
  waiting: { color: COLORS.TEXT_GHOST, fontSize: 24, textAlign: 'center' },

  dots: { flexDirection: 'row', justifyContent: 'center', gap: 8, marginBottom: 16 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  dotDone: { backgroundColor: COLORS.RED },
  dotActive: { backgroundColor: COLORS.RED, opacity: 0.5 },
  dotPending: { backgroundColor: COLORS.TEXT_GHOST },

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
