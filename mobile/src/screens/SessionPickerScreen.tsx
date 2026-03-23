import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { COLORS, TYPOGRAPHY, SPACING } from '../theme';
import { generatePlan, TodayData, ProgramDaySummary } from '../api/session';

type Props = NativeStackScreenProps<any, 'SessionPicker'>;

export default function SessionPickerScreen({ route, navigation }: Props) {
  const today = route.params?.today as TodayData;
  const [selectedDay, setSelectedDay] = useState<ProgramDaySummary | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    const rec = today.program_days.find((d) => d.day_number === today.day_number) ?? null;
    setSelectedDay(rec);
  }, [today]);

  const handleStart = async () => {
    if (starting || !selectedDay) return;
    setStarting(true);
    try {
      const plan = await generatePlan({ program_day_id: selectedDay.id });
      navigation.replace('ActiveSession', { plan, today });
    } catch (e) {
      console.error('Plan generation failed:', e);
    } finally {
      setStarting(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <Text style={styles.header}>세션 선택</Text>

      {/* Day Cards */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.pickerRow}
        style={styles.pickerScroll}
      >
        {today.program_days.map((day) => {
          const isSelected = selectedDay?.day_number === day.day_number;
          const isRecommended = day.day_number === today.day_number;
          return (
            <TouchableOpacity
              key={day.day_number}
              style={[styles.dayCard, isSelected && styles.dayCardSelected]}
              onPress={() => setSelectedDay(day)}
              activeOpacity={0.7}
            >
              {isRecommended && <View style={styles.recommendedDot} />}
              <Text style={[styles.dayNumber, isSelected && styles.dayNumberSelected]}>
                {day.day_number}
              </Text>
              <Text
                style={[styles.dayTheme, isSelected && styles.dayThemeSelected]}
                numberOfLines={2}
              >
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
    paddingTop: 20,
    paddingBottom: 50,
  },
  header: {
    ...TYPOGRAPHY.TITLE,
    color: COLORS.TEXT_1,
    textAlign: 'center',
    marginBottom: 28,
  },
  pickerScroll: {
    maxHeight: 100,
    marginBottom: 24,
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
