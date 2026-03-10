import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import * as Speech from 'expo-speech';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { logSession, PlanResponse } from '../api/session';

type Props = NativeStackScreenProps<any, 'ActiveSession'>;

export default function ActiveSessionScreen({ route, navigation }: Props) {
  const { planId, plan: planDetail } = route.params as {
    planId: string;
    plan: PlanResponse['plan'];
  };

  const rounds = planDetail.rounds;
  const [pmin, pmax] = planDetail.pace_interval_sec ?? [3, 5];

  const [phase, setPhase] = useState<'round' | 'rest'>('round');
  const [roundNum, setRoundNum] = useState(rounds[0]?.round_number ?? 1);
  const [secondsLeft, setSecondsLeft] = useState(rounds[0]?.duration_seconds ?? 0);
  const [currentCombo, setCurrentCombo] = useState('');
  const [currentActions, setCurrentActions] = useState<string[]>([]);

  const abortRef = useRef(false);
  const roundsCompletedRef = useRef(0);
  const combosRef = useRef(0);
  const startedAtRef = useRef(new Date());

  const speak = (text: string) => {
    Speech.speak(text, { language: 'ko-KR' });
  };

  const sleep = (ms: number) =>
    new Promise<void>((resolve) => setTimeout(resolve, ms));

  useEffect(() => {
    let countdownTimer: ReturnType<typeof setInterval> | null = null;

    const runSession = async () => {
      for (let i = 0; i < rounds.length; i++) {
        if (abortRef.current) break;

        const round = rounds[i];
        let secsLeft = round.duration_seconds;

        setRoundNum(round.round_number);
        setPhase('round');
        setCurrentCombo('');
        setCurrentActions([]);
        setSecondsLeft(secsLeft);
        speak(`라운드 ${round.round_number}`);

        // Countdown timer for this round
        if (countdownTimer) clearInterval(countdownTimer);
        countdownTimer = setInterval(() => {
          secsLeft = Math.max(0, secsLeft - 1);
          setSecondsLeft(secsLeft);
        }, 1000);

        // 3s lead-in before first combo
        await sleep(3000);

        // Deliver combos until round ends
        const roundEnd = Date.now() + (round.duration_seconds - 3) * 1000;
        let comboIdx = 0;

        while (Date.now() < roundEnd && !abortRef.current) {
          const instr = round.instructions[comboIdx % round.instructions.length];
          comboIdx++;
          combosRef.current++;
          setCurrentCombo(instr.combo_display_name);
          setCurrentActions(instr.actions);
          speak(instr.combo_display_name);

          const interval = (pmin + Math.random() * (pmax - pmin)) * 1000;
          const waitUntil = Math.min(Date.now() + interval, roundEnd);
          await sleep(Math.max(0, waitUntil - Date.now()));
        }

        if (countdownTimer) clearInterval(countdownTimer);
        roundsCompletedRef.current++;
        setCurrentCombo('');

        if (abortRef.current) break;

        // Rest period (skip after last round)
        const isLastRound = i === rounds.length - 1;
        if (!isLastRound && round.rest_after_seconds > 0) {
          let restSecs = round.rest_after_seconds;
          setPhase('rest');
          setSecondsLeft(restSecs);
          speak('휴식');

          countdownTimer = setInterval(() => {
            restSecs = Math.max(0, restSecs - 1);
            setSecondsLeft(restSecs);
          }, 1000);

          await sleep(round.rest_after_seconds * 1000);
          if (countdownTimer) clearInterval(countdownTimer);
        }
      }

      if (countdownTimer) clearInterval(countdownTimer);
      Speech.stop();

      const completedAt = new Date();
      const durationSec = (completedAt.getTime() - startedAtRef.current.getTime()) / 1000;
      const status = abortRef.current ? 'abandoned' : 'completed';

      // Save result to server (fire and forget)
      logSession({
        drill_plan_id: planId,
        template_name: planDetail.template,
        started_at: startedAtRef.current.toISOString(),
        completed_at: completedAt.toISOString(),
        total_duration_sec: durationSec,
        rounds_completed: roundsCompletedRef.current,
        rounds_total: rounds.length,
        combos_delivered: combosRef.current,
        status,
      }).catch(() => {}); // ignore network errors

      navigation.replace('SessionEnd', {
        status,
        rounds: roundsCompletedRef.current,
        combos: combosRef.current,
        duration: Math.round(durationSec),
      });
    };

    runSession();

    return () => {
      if (countdownTimer) clearInterval(countdownTimer);
      Speech.stop();
    };
  }, []);

  const handleAbort = () => {
    Alert.alert('세션 중단', '정말 중단하시겠습니까?', [
      { text: '계속', style: 'cancel' },
      {
        text: '중단',
        style: 'destructive',
        onPress: () => {
          abortRef.current = true;
        },
      },
    ]);
  };

  const mm = String(Math.floor(secondsLeft / 60)).padStart(2, '0');
  const ss = String(secondsLeft % 60).padStart(2, '0');

  return (
    <View style={styles.container}>
      {/* Round info */}
      <View style={styles.header}>
        <Text style={styles.roundLabel}>
          {phase === 'rest' ? '휴식' : `Round ${roundNum} / ${rounds.length}`}
        </Text>
        <Text style={styles.timer}>{mm}:{ss}</Text>
      </View>

      {/* Round progress dots */}
      <View style={styles.dots}>
        {rounds.map((_, i) => (
          <View
            key={i}
            style={[
              styles.dot,
              i < roundsCompletedRef.current
                ? styles.dotDone
                : i === roundNum - 1
                ? styles.dotActive
                : styles.dotPending,
            ]}
          />
        ))}
      </View>

      {/* Current combo */}
      <View style={styles.comboContainer}>
        {currentCombo ? (
          <>
            <Text style={styles.comboName}>{currentCombo}</Text>
            <Text style={styles.comboActions}>{currentActions.join(' → ')}</Text>
          </>
        ) : (
          <Text style={styles.waiting}>
            {phase === 'rest' ? '잠시 쉬세요' : '준비...'}
          </Text>
        )}
      </View>

      {/* Abort button */}
      <TouchableOpacity style={styles.abortButton} onPress={handleAbort}>
        <Text style={styles.abortText}>중단</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a', padding: 24, justifyContent: 'space-between' },
  header: { alignItems: 'center', marginTop: 40 },
  roundLabel: { color: '#888', fontSize: 14, textTransform: 'uppercase', letterSpacing: 2 },
  timer: { color: '#fff', fontSize: 72, fontWeight: '200', letterSpacing: 4, marginTop: 8 },
  dots: { flexDirection: 'row', justifyContent: 'center', gap: 8 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  dotDone: { backgroundColor: '#e63946' },
  dotActive: { backgroundColor: '#e63946', opacity: 0.5 },
  dotPending: { backgroundColor: '#333' },
  comboContainer: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  comboName: { color: '#fff', fontSize: 48, fontWeight: '700', textAlign: 'center', marginBottom: 16 },
  comboActions: { color: '#666', fontSize: 18, textAlign: 'center' },
  waiting: { color: '#333', fontSize: 24 },
  abortButton: {
    alignSelf: 'center',
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    marginBottom: 24,
  },
  abortText: { color: '#666', fontSize: 16 },
});
