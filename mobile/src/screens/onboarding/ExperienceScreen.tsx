import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { COLORS, SPACING, TYPOGRAPHY } from '../../theme';
import SectionLabel from '../../components/SectionLabel';
import PrimaryButton from '../../components/PrimaryButton';

type Props = NativeStackScreenProps<any, 'Experience'>;

const OPTIONS = [
  { key: 'beginner', label: '처음', desc: '복싱을 배운 적 없음' },
  { key: 'novice', label: '초급', desc: '기본 펀치를 알고 있음' },
  { key: 'intermediate', label: '중급', desc: '콤비네이션 가능' },
  { key: 'advanced', label: '상급', desc: '스파링 경험 있음' },
];

export default function ExperienceScreen({ navigation }: Props) {
  const [selected, setSelected] = useState<string | null>(null);

  const handleSelect = (key: string) => {
    setSelected(key);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>ATOM</Text>
        <Text style={styles.subtitle}>AI BOXING COACH</Text>
      </View>

      <View style={styles.divider} />

      <SectionLabel text="복싱 경험" />

      <View style={styles.options}>
        {OPTIONS.map((opt) => {
          const isSelected = selected === opt.key;
          return (
            <TouchableOpacity
              key={opt.key}
              style={[styles.option, isSelected ? styles.optionSelected : styles.optionDefault]}
              onPress={() => handleSelect(opt.key)}
              activeOpacity={0.7}
            >
              <Text style={[styles.optionLabel, isSelected && { color: COLORS.RED }]}>
                {opt.label}
              </Text>
              <Text style={styles.optionDesc}>{opt.desc}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <View style={styles.bottom}>
        <PrimaryButton
          label="다음"
          onPress={() => navigation.navigate('Environment', { experience_level: selected })}
          disabled={!selected}
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
  },
  header: { marginTop: 60, marginBottom: 24 },
  title: {
    color: COLORS.RED,
    fontSize: 42,
    fontWeight: '900',
    letterSpacing: 6,
  },
  subtitle: {
    color: COLORS.TEXT_3,
    fontSize: TYPOGRAPHY.SECTION_LABEL.fontSize,
    letterSpacing: 3,
    marginTop: 4,
  },
  divider: { height: 1, backgroundColor: COLORS.BORDER, marginBottom: SPACING.GAP_SECTION },

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
    marginBottom: 2,
  },
  optionDesc: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize },

  bottom: { flex: 1, justifyContent: 'flex-end', paddingBottom: 32 },
});
