import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, TYPOGRAPHY, SPACING } from '../../theme';
import { apiFetch } from '../../api/client';

const OPTIONS = [
  { value: 'basics', label: '기본기', desc: '폼과 정확성에 집중' },
  { value: 'cardio', label: '체력', desc: '심박수와 지구력 위주' },
  { value: 'speed', label: '스피드', desc: '빠른 손과 반응속도' },
  { value: 'all', label: '다 좋아', desc: '골고루 훈련할게요' },
];

interface Props {
  route?: any;
  onComplete: () => void;
}

export default function OnboardingPreferenceScreen({ route, onComplete }: Props) {
  const experience = route?.params?.experience || 'beginner';

  const handleSelect = async (preference: string) => {
    try {
      // Update profile with experience + preference
      await apiFetch('/api/profile', {
        method: 'PUT',
        body: JSON.stringify({
          experience_level: experience === 'novice' ? 'beginner' : experience,
          training_preference: preference,
        }),
      });
    } catch (e) {
      console.warn('Profile update failed:', e);
    }
    onComplete();
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.step}>2/2</Text>
        <Text style={styles.title}>어떤 훈련을 하고 싶어?</Text>
      </View>
      <View style={styles.options}>
        {OPTIONS.map((opt) => (
          <TouchableOpacity
            key={opt.value}
            style={styles.option}
            onPress={() => handleSelect(opt.value)}
          >
            <Text style={styles.optionLabel}>{opt.label}</Text>
            <Text style={styles.optionDesc}>{opt.desc}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    paddingHorizontal: SPACING.PADDING_SCREEN,
    paddingTop: 100,
  },
  header: {
    marginBottom: 40,
  },
  step: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_3,
    marginBottom: 8,
  },
  title: {
    ...TYPOGRAPHY.TITLE,
    color: COLORS.TEXT_1,
  },
  options: {
    gap: 12,
  },
  option: {
    backgroundColor: COLORS.SURFACE,
    borderRadius: SPACING.RADIUS_CARD,
    padding: 20,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  optionLabel: {
    ...TYPOGRAPHY.CARD_TITLE,
    color: COLORS.TEXT_1,
    marginBottom: 4,
  },
  optionDesc: {
    ...TYPOGRAPHY.META,
    color: COLORS.TEXT_2,
  },
});
