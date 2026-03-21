import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { COLORS, SPACING, TYPOGRAPHY } from '../../theme';
import SectionLabel from '../../components/SectionLabel';
import PrimaryButton from '../../components/PrimaryButton';
import { updateProfile } from '../../api/profile';

type Props = NativeStackScreenProps<any, 'Frequency'> & {
  onComplete: () => void;
};

const OPTIONS = [
  { key: 'daily', label: '매일', desc: '주 6-7회' },
  { key: '5_per_week', label: '주 5회', desc: '평일 매일' },
  { key: '3_per_week', label: '주 3회', desc: '이틀 간격' },
  { key: '1_per_week', label: '주 1회', desc: '주말에 한 번' },
];

export default function FrequencyScreen({ route, onComplete }: Props) {
  const params = route.params as {
    experience_level: string;
    training_environment: string;
    equipment: string[];
    goals: string[];
  };
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSelect = (key: string) => {
    setSelected(key);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleSubmit = async () => {
    if (!selected) return;
    setLoading(true);
    try {
      await updateProfile({
        experience_level: params.experience_level,
        training_environment: params.training_environment,
        equipment: params.equipment,
        goals: params.goals,
        training_frequency: selected,
      });
    } catch {
      // proceed even if API fails
    }
    await onComplete();
  };

  return (
    <View style={styles.container}>
      <SectionLabel text="훈련 빈도" />

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
          label="시작하기"
          onPress={handleSubmit}
          disabled={!selected || loading}
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
