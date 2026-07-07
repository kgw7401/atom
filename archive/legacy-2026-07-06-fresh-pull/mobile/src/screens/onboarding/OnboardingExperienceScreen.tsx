import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, TYPOGRAPHY, SPACING } from '../../theme';

const OPTIONS = [
  { value: 'beginner', label: '처음이야', desc: '복싱을 해본 적 없어요' },
  { value: 'novice', label: '조금 해봤어', desc: '기본 펀치는 알아요' },
  { value: 'intermediate', label: '꽤 쳤어', desc: '콤비네이션도 할 수 있어요' },
];

export default function OnboardingExperienceScreen({ navigation }: any) {
  const handleSelect = (value: string) => {
    // Store selection for later profile update
    navigation.navigate('Preference', { experience: value });
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.step}>1/2</Text>
        <Text style={styles.title}>복싱 경험이 있나요?</Text>
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
