import React, { useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, TYPOGRAPHY, SPACING } from '../theme';
import { fetchToday, generatePlan, TodayData, ProgramDaySummary } from '../api/session';
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
  const [selectedDay, setSelectedDay] = useState<ProgramDaySummary | null>(null);
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
              const rec = data.program_days.find((d) => d.day_number === data.day_number) ?? null;
              setSelectedDay(rec);
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
        .then((data) => {
          if (!cancelled) {
            setToday(data);
            const rec = data.program_days.find((d) => d.day_number === data.day_number) ?? null;
            setSelectedDay(rec);
          }
        })
        .catch((e) => console.warn('fetchToday failed:', e))
        .finally(() => { if (!cancelled) setLoading(false); });
      return () => { cancelled = true; };
    }, [])
  );

  const handleStart = async () => {
    if (starting) return;
    setStarting(true);
    try {
      const plan = await generatePlan(selectedDay ? { program_day_id: selectedDay.id } : {});
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

      {/* Session Picker */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.pickerRow}
        style={styles.pickerScroll}
      >
        {(today?.program_days ?? []).map((day) => {
          const isSelected = selectedDay?.day_number === day.day_number;
          const isRecommended = day.day_number === today?.day_number;
          return (
            <TouchableOpacity
              key={day.day_number}
              style={[
                styles.dayCard,
                isSelected && styles.dayCardSelected,
              ]}
              onPress={() => setSelectedDay(day)}
              activeOpacity={0.7}
            >
              {isRecommended && <View style={styles.recommendedDot} />}
              <Text style={[styles.dayNumber, isSelected && styles.dayNumberSelected]}>
                {day.day_number}
              </Text>
              <Text style={[styles.dayTheme, isSelected && styles.dayThemeSelected]} numberOfLines={2}>
                {day.theme}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      {/* Selected Day Info */}
      {selectedDay && (
        <View style={styles.selectedInfo}>
          <Text style={styles.themeTitle}>{selectedDay.theme}</Text>
          <Text style={styles.themeDesc}>{selectedDay.theme_description}</Text>
        </View>
      )}

      {/* Coach Comment */}
      {selectedDay?.coach_comment ? (
        <View style={styles.coachCard}>
          <Text style={styles.coachLabel}>코치</Text>
          <Text style={styles.coachText}>{selectedDay.coach_comment}</Text>
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
  pickerScroll: {
    maxHeight: 100,
    marginBottom: 20,
  },
  pickerRow: {
    gap: 10,
    paddingHorizontal: 2,
  },
  dayCard: {
    width: 72,
    height: 80,
    backgroundColor: COLORS.SURFACE,
    borderRadius: SPACING.RADIUS_CARD,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 4,
  },
  dayCardSelected: {
    backgroundColor: COLORS.RED_BG,
    borderColor: COLORS.RED,
    borderWidth: 1.5,
  },
  recommendedDot: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 5,
    height: 5,
    borderRadius: 2.5,
    backgroundColor: COLORS.RED,
  },
  dayNumber: {
    fontSize: 20,
    fontWeight: '700',
    color: COLORS.TEXT_3,
    marginBottom: 4,
  },
  dayNumberSelected: {
    color: COLORS.RED,
  },
  dayTheme: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_3,
    textAlign: 'center',
  },
  dayThemeSelected: {
    color: COLORS.TEXT_1,
  },
  selectedInfo: {
    alignItems: 'center',
    marginBottom: 16,
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
