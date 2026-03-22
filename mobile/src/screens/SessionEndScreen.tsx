import React, { useEffect, useRef, useState } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Animated, { FadeIn, FadeInDown, ZoomIn } from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { logSession, SessionLogRequest, TodayData, fetchToday } from '../api/session';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';

type Props = NativeStackScreenProps<any, 'SessionEnd'>;

export default function SessionEndScreen({ route, navigation }: Props) {
  const {
    status,
    rounds,
    roundsTotal,
    segments,
    duration,
    today: initialToday,
    planResponse,
    logPayload,
  } = route.params ?? {};

  const logged = useRef(false);
  const [todayData, setTodayData] = useState<TodayData | null>(initialToday);

  useEffect(() => {
    if (logged.current || !logPayload) return;
    logged.current = true;
    const body: SessionLogRequest = { ...logPayload };
    logSession(body)
      .then(() => {
        // Refresh today data to get updated streak
        fetchToday()
          .then(setTodayData)
          .catch(() => {});
      })
      .catch(() => {});
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, [logPayload]);

  const isCompleted = status === 'completed';
  const streak = todayData?.streak ?? 0;
  const dayNumber = todayData?.day_number ?? 1;
  const dayTotal = todayData?.day_total ?? 7;
  const coachComment = planResponse?.coach_comment || todayData?.coach_comment || '';
  const nextPreview = todayData?.next_day_preview;
  const theme = planResponse?.theme || todayData?.theme || '';

  const durationMin = Math.floor((duration ?? 0) / 60);
  const durationSec = (duration ?? 0) % 60;
  const durationStr = durationMin > 0 ? `${durationMin}분 ${durationSec}초` : `${durationSec}초`;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Checkmark */}
      <Animated.View
        entering={ZoomIn.springify().damping(12)}
        style={[styles.statusCircle, isCompleted ? styles.statusDone : styles.statusAbandoned]}
      >
        <Text style={[styles.statusIcon, { color: isCompleted ? COLORS.GOLD : COLORS.ORANGE }]}>
          {isCompleted ? '✓' : '–'}
        </Text>
      </Animated.View>

      {/* Title */}
      <Animated.Text entering={FadeIn.delay(200)} style={styles.title}>
        {isCompleted ? '완료!' : '중단되었습니다.'}
      </Animated.Text>

      {/* Coach Comment */}
      {isCompleted && coachComment ? (
        <Animated.View entering={FadeInDown.delay(400)} style={styles.coachCard}>
          <Text style={styles.coachLabel}>코치</Text>
          <Text style={styles.coachText}>{coachComment}</Text>
        </Animated.View>
      ) : null}

      {/* Streak */}
      {isCompleted && streak > 0 ? (
        <Animated.View entering={FadeInDown.delay(600)} style={styles.streakSection}>
          <Text style={styles.streakNumber}>{streak}</Text>
          <Text style={styles.streakLabel}>일 연속</Text>
        </Animated.View>
      ) : null}

      {/* Day Progress */}
      {isCompleted ? (
        <Animated.View entering={FadeInDown.delay(800)} style={styles.progressSection}>
          <View style={styles.dots}>
            {Array.from({ length: dayTotal }, (_, i) => (
              <View
                key={i}
                style={[
                  styles.dot,
                  i < dayNumber && styles.dotCompleted,
                  i >= dayNumber && styles.dotPending,
                ]}
              />
            ))}
          </View>
          <Text style={styles.dayLabel}>Day {dayNumber} / {dayTotal} 완료</Text>
        </Animated.View>
      ) : null}

      {/* Session Stats */}
      <Animated.View entering={FadeInDown.delay(1000)} style={styles.statsRow}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>3R</Text>
          <Text style={styles.statLabel}>라운드</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{durationStr}</Text>
          <Text style={styles.statLabel}>시간</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{segments ?? 0}</Text>
          <Text style={styles.statLabel}>콤보</Text>
        </View>
      </Animated.View>

      {/* Next Day Preview */}
      {isCompleted && nextPreview ? (
        <Animated.View entering={FadeInDown.delay(1200)} style={styles.nextCard}>
          <Text style={styles.nextLabel}>내일</Text>
          <Text style={styles.nextText}>
            Day {nextPreview.day_number} — {nextPreview.theme}
          </Text>
        </Animated.View>
      ) : null}

      {/* Home Button */}
      <Animated.View entering={FadeIn.delay(1400)} style={styles.actions}>
        <TouchableOpacity
          style={styles.homeBtn}
          onPress={() => navigation.popToTop()}
          activeOpacity={0.85}
        >
          <Text style={styles.homeBtnText}>홈으로 돌아가기</Text>
        </TouchableOpacity>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.BG },
  content: {
    alignItems: 'center',
    padding: SPACING.PADDING_SCREEN,
    paddingTop: 80,
    paddingBottom: 60,
  },

  // Status
  statusCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  statusDone: { backgroundColor: '#1a1a0a', borderWidth: 2, borderColor: COLORS.GOLD },
  statusAbandoned: { backgroundColor: '#2e2a1a', borderWidth: 2, borderColor: COLORS.ORANGE },
  statusIcon: { fontSize: 36, fontWeight: '700' },

  // Title
  title: {
    ...TYPOGRAPHY.TITLE,
    color: COLORS.TEXT_1,
    marginBottom: 24,
  },

  // Coach
  coachCard: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: SPACING.RADIUS_CARD,
    padding: SPACING.PADDING_CARD,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    width: '100%',
    marginBottom: 24,
  },
  coachLabel: {
    ...TYPOGRAPHY.SECTION_LABEL,
    color: COLORS.BLUE,
    marginBottom: 8,
  },
  coachText: {
    ...TYPOGRAPHY.COACH_BODY,
    color: COLORS.TEXT_1,
  },

  // Streak
  streakSection: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 20,
  },
  streakNumber: {
    ...TYPOGRAPHY.STREAK_NUMBER,
    color: COLORS.ORANGE,
  },
  streakLabel: {
    ...TYPOGRAPHY.BODY,
    color: COLORS.ORANGE,
    marginLeft: 4,
  },

  // Progress
  progressSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  dots: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 8,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  dotCompleted: {
    backgroundColor: COLORS.RED,
  },
  dotPending: {
    backgroundColor: COLORS.SURFACE,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  dayLabel: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_2,
  },

  // Stats
  statsRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.SURFACE,
    borderRadius: SPACING.RADIUS_CARD,
    padding: 16,
    width: '100%',
    marginBottom: 24,
    alignItems: 'center',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    ...TYPOGRAPHY.STAT_VALUE,
    color: COLORS.TEXT_1,
  },
  statLabel: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_3,
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: COLORS.BORDER,
  },

  // Next day
  nextCard: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: SPACING.RADIUS_CARD,
    padding: SPACING.PADDING_CARD,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    width: '100%',
    marginBottom: 24,
    alignItems: 'center',
  },
  nextLabel: {
    ...TYPOGRAPHY.SECTION_LABEL,
    color: COLORS.TEXT_3,
    marginBottom: 4,
  },
  nextText: {
    ...TYPOGRAPHY.CARD_TITLE,
    color: COLORS.TEXT_1,
  },

  // Actions
  actions: { width: '100%', marginTop: 8 },
  homeBtn: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: 12,
    paddingVertical: 18,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  homeBtnText: {
    color: COLORS.TEXT_1,
    ...TYPOGRAPHY.CARD_TITLE,
  },
});
