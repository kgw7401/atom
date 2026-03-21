import React, { useEffect, useRef } from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Animated, { FadeIn, ZoomIn } from 'react-native-reanimated';
import { logSession, SessionLogRequest } from '../api/session';
import { COLORS, SPACING } from '../theme';
import StatCard from '../components/StatCard';
import PrimaryButton from '../components/PrimaryButton';
import SecondaryButton from '../components/SecondaryButton';

type Props = NativeStackScreenProps<any, 'SessionEnd'>;

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m === 0) return `${s}초`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function SessionEndScreen({ route, navigation }: Props) {
  const {
    status,
    rounds,
    roundsTotal,
    segments,
    duration,
    logPayload,
  } = route.params ?? {};

  const logged = useRef(false);

  useEffect(() => {
    if (logged.current || !logPayload) return;
    logged.current = true;
    const body: SessionLogRequest = { ...logPayload };
    logSession(body).catch(() => {});
  }, [logPayload]);

  const isCompleted = status === 'completed';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View
        entering={ZoomIn.springify().damping(12)}
        style={[styles.statusCircle, isCompleted ? styles.statusDone : styles.statusAbandoned]}
      >
        <Text style={[styles.statusIcon, { color: isCompleted ? COLORS.GREEN : COLORS.ORANGE }]}>
          {isCompleted ? '✓' : '–'}
        </Text>
      </Animated.View>

      <Animated.Text entering={FadeIn.delay(200)} style={styles.title}>
        {isCompleted ? '수고하셨습니다.' : '중단되었습니다.'}
      </Animated.Text>

      {/* Stats */}
      <Animated.View entering={FadeIn.delay(400)} style={styles.statsRow}>
        <StatCard
          value={isCompleted ? String(rounds ?? 0) : `${rounds ?? 0}/${roundsTotal ?? 0}`}
          label="라운드"
        />
        <StatCard value={String(segments ?? 0)} label="구간" />
        <StatCard value={formatDuration(duration ?? 0)} label="시간" />
      </Animated.View>

      {/* Actions */}
      <View style={styles.actions}>
        <PrimaryButton
          label={isCompleted ? '다시 시작' : '다시 시도'}
          onPress={() => navigation.navigate('SessionSetup')}
        />
        <View style={{ height: 12 }} />
        <SecondaryButton label="홈으로" onPress={() => navigation.navigate('Home')} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.BG },
  content: {
    alignItems: 'center',
    padding: 32,
    paddingTop: 60,
    paddingBottom: 48,
  },

  statusCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  statusDone: { backgroundColor: '#0a1f0a', borderWidth: 2, borderColor: COLORS.GREEN },
  statusAbandoned: { backgroundColor: '#2e2a1a', borderWidth: 2, borderColor: COLORS.ORANGE },
  statusIcon: { fontSize: 36, fontWeight: '700' },

  title: {
    color: COLORS.TEXT_1,
    fontSize: 28,
    fontWeight: '700',
    marginBottom: SPACING.GAP_SECTION,
  },

  statsRow: {
    flexDirection: 'row',
    gap: 10,
    width: '100%',
    marginBottom: 16,
  },

  actions: { width: '100%', marginTop: 8 },
});
