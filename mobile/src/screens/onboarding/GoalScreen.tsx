import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { COLORS, SPACING, TYPOGRAPHY } from '../../theme';
import SectionLabel from '../../components/SectionLabel';
import PrimaryButton from '../../components/PrimaryButton';

type Props = NativeStackScreenProps<any, 'Goal'>;

const GOALS = [
  { key: 'speed', label: '스피드 향상' },
  { key: 'power', label: '파워 강화' },
  { key: 'conditioning', label: '체력/컨디셔닝' },
  { key: 'defense', label: '방어 기술' },
  { key: 'stress', label: '스트레스 해소' },
  { key: 'weight', label: '다이어트' },
  { key: 'technique', label: '실전 기술' },
];

export default function GoalScreen({ route, navigation }: Props) {
  const params = route.params as Record<string, any>;
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const handleToggle = (key: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <View style={styles.container}>
      <SectionLabel text="훈련 목표" />
      <Text style={styles.hint}>여러 개 선택 가능</Text>

      <View style={styles.options}>
        {GOALS.map((goal) => {
          const isSelected = selected.has(goal.key);
          return (
            <TouchableOpacity
              key={goal.key}
              style={[styles.option, isSelected ? styles.optionSelected : styles.optionDefault]}
              onPress={() => handleToggle(goal.key)}
              activeOpacity={0.7}
            >
              <Text style={[styles.optionLabel, isSelected && { color: COLORS.RED }]}>
                {goal.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <View style={styles.bottom}>
        <PrimaryButton
          label="다음"
          onPress={() =>
            navigation.navigate('Frequency', {
              ...params,
              goals: Array.from(selected),
            })
          }
          disabled={selected.size === 0}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    padding: SPACING.PADDING_SCREEN,
    paddingTop: 16,
  },
  hint: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize, marginBottom: 12 },
  options: { gap: SPACING.GAP_ITEM },
  option: {
    borderWidth: 1,
    borderRadius: SPACING.RADIUS_CARD,
    padding: SPACING.PADDING_CARD,
  },
  optionDefault: {
    backgroundColor: COLORS.SURFACE,
    borderColor: COLORS.BORDER,
  },
  optionSelected: {
    backgroundColor: COLORS.RED_BG,
    borderColor: COLORS.RED,
  },
  optionLabel: {
    color: COLORS.TEXT_1,
    fontSize: 16,
    fontWeight: '600',
  },
  bottom: { flex: 1, justifyContent: 'flex-end', paddingBottom: 32 },
});
