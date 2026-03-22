import React, { useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, TYPOGRAPHY, SPACING } from '../theme';
import { fetchToday, generatePlan, TodayData } from '../api/session';

type Props = { navigation: NativeStackNavigationProp<any> };

export default function HomeScreen({ navigation }: Props) {
  const [today, setToday] = useState<TodayData | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);

  useFocusEffect(
    useCallback(() => {
      let cancelled = false;
      setLoading(true);
      fetchToday()
        .then((data) => { if (!cancelled) setToday(data); })
        .catch((e) => console.warn('fetchToday failed:', e))
        .finally(() => { if (!cancelled) setLoading(false); });
      return () => { cancelled = true; };
    }, [])
  );

  const handleStart = async () => {
    if (starting) return;
    setStarting(true);
    try {
      const plan = await generatePlan({});
      navigation.navigate('ActiveSession', { plan, today });
    } catch (e) {
      console.error('Plan generation failed:', e);
    } finally {
      setStarting(false);
    }
  };

  if (loading) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator color={COLORS.RED} size="large" />
      </View>
    );
  }

  const streak = today?.streak ?? 0;
  const dayNumber = today?.day_number ?? 1;
  const dayTotal = today?.day_total ?? 7;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.logo}>ATOM</Text>
        {streak > 0 && (
          <View style={styles.streakBadge}>
            <Text style={styles.streakText}>{streak}일 연속</Text>
          </View>
        )}
      </View>

      {/* Day Progress */}
      <View style={styles.progressSection}>
        <Text style={styles.sectionLabel}>WEEK {today?.week ?? 1}</Text>
        <View style={styles.dots}>
          {Array.from({ length: dayTotal }, (_, i) => (
            <View
              key={i}
              style={[
                styles.dot,
                i < dayNumber - 1 && styles.dotCompleted,
                i === dayNumber - 1 && styles.dotCurrent,
              ]}
            />
          ))}
        </View>
        <Text style={styles.dayLabel}>Day {dayNumber} / {dayTotal}</Text>
      </View>

      {/* Today's Theme */}
      <View style={styles.themeSection}>
        <Text style={styles.themeTitle}>{today?.theme ?? '훈련'}</Text>
        <Text style={styles.themeDesc}>{today?.theme_description ?? ''}</Text>
      </View>

      {/* Coach Comment */}
      {today?.coach_comment ? (
        <View style={styles.coachCard}>
          <Text style={styles.coachLabel}>코치</Text>
          <Text style={styles.coachText}>{today.coach_comment}</Text>
        </View>
      ) : null}

      {/* Spacer */}
      <View style={{ flex: 1 }} />

      {/* Next Day Preview */}
      {today?.next_day_preview && (
        <Text style={styles.nextPreview}>
          내일: Day {today.next_day_preview.day_number} — {today.next_day_preview.theme}
        </Text>
      )}

      {/* Start Button */}
      <TouchableOpacity
        style={[styles.startBtn, starting && { opacity: 0.6 }]}
        onPress={handleStart}
        activeOpacity={0.85}
        disabled={starting}
      >
        {starting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.startBtnText}>시작</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    paddingHorizontal: SPACING.PADDING_SCREEN,
    paddingTop: 70,
    paddingBottom: 50,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 32,
  },
  logo: {
    ...TYPOGRAPHY.APP_TITLE,
    color: COLORS.RED,
  },
  streakBadge: {
    backgroundColor: COLORS.SURFACE,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: SPACING.RADIUS_BADGE,
    borderWidth: 1,
    borderColor: COLORS.ORANGE,
  },
  streakText: {
    color: COLORS.ORANGE,
    ...TYPOGRAPHY.META,
    fontWeight: '700',
  },
  progressSection: {
    alignItems: 'center',
    marginBottom: 32,
  },
  sectionLabel: {
    ...TYPOGRAPHY.SECTION_LABEL,
    color: COLORS.TEXT_3,
    marginBottom: 12,
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
    backgroundColor: COLORS.SURFACE,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  dotCompleted: {
    backgroundColor: COLORS.RED,
    borderColor: COLORS.RED,
  },
  dotCurrent: {
    backgroundColor: COLORS.BG,
    borderColor: COLORS.RED,
    borderWidth: 2,
  },
  dayLabel: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_2,
  },
  themeSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  themeTitle: {
    ...TYPOGRAPHY.TITLE,
    color: COLORS.TEXT_1,
    marginBottom: 8,
  },
  themeDesc: {
    ...TYPOGRAPHY.BODY,
    color: COLORS.TEXT_2,
    textAlign: 'center',
  },
  coachCard: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: SPACING.RADIUS_CARD,
    padding: SPACING.PADDING_CARD,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
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
  nextPreview: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_3,
    textAlign: 'center',
    marginBottom: 16,
  },
  startBtn: {
    backgroundColor: COLORS.RED,
    borderRadius: 16,
    paddingVertical: 20,
    alignItems: 'center',
    width: '85%',
    alignSelf: 'center',
  },
  startBtnText: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: 2,
  },
});
