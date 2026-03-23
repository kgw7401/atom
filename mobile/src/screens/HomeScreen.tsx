import React, { useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, TYPOGRAPHY, SPACING } from '../theme';
import { fetchToday, generatePlan, TodayData } from '../api/session';
import { updateProfile } from '../api/profile';

type Props = { navigation: NativeStackNavigationProp<any> };

const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
const LEVEL_LABELS: Record<string, string> = {
  beginner: '입문',
  novice: '입문',
  intermediate: '중급',
  advanced: '고급',
};

export default function HomeScreen({ navigation }: Props) {
  const [today, setToday] = useState<TodayData | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);

  const handleLevelChange = () => {
    const raw = today?.level ?? 'beginner';
    const current = raw === 'novice' ? 'beginner' : raw;
    const options = LEVELS.filter((l) => l !== current);
    Alert.alert(
      '레벨 변경',
      `현재: ${LEVEL_LABELS[current]}`,
      [
        ...options.map((lvl) => ({
          text: LEVEL_LABELS[lvl],
          onPress: async () => {
            try {
              await updateProfile({ experience_level: lvl });
              const data = await fetchToday();
              setToday(data);
            } catch (e) {
              console.error('Level change failed:', e);
            }
          },
        })),
        { text: '취소', style: 'cancel' as const },
      ],
    );
  };

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

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.logo}>ATOM</Text>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.levelBadge} onPress={handleLevelChange}>
            <Text style={styles.levelText}>
              {LEVEL_LABELS[today?.level ?? 'beginner']}
            </Text>
          </TouchableOpacity>
          {streak > 0 && (
            <View style={styles.streakBadge}>
              <Text style={styles.streakText}>{streak}일 연속</Text>
            </View>
          )}
        </View>
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
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  levelBadge: {
    backgroundColor: COLORS.SURFACE,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: SPACING.RADIUS_BADGE,
    borderWidth: 1,
    borderColor: COLORS.TEXT_3,
  },
  levelText: {
    color: COLORS.TEXT_2,
    ...TYPOGRAPHY.META,
    fontWeight: '700',
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
