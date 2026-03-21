import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { COLORS, SPACING, TYPOGRAPHY } from '../../theme';
import SectionLabel from '../../components/SectionLabel';
import PrimaryButton from '../../components/PrimaryButton';

type Props = NativeStackScreenProps<any, 'Environment'>;

const OPTIONS = [
  { key: 'home', label: '집', desc: '집에서 혼자 훈련' },
  { key: 'gym', label: '체육관', desc: '체육관에서 훈련' },
];

export default function EnvironmentScreen({ route, navigation }: Props) {
  const params = route.params as { experience_level: string };
  const [selected, setSelected] = useState<string | null>(null);

  const handleSelect = (key: string) => {
    setSelected(key);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  return (
    <View style={styles.container}>
      <SectionLabel text="훈련 환경" />

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
          onPress={() =>
            navigation.navigate('Equipment', {
              ...params,
              training_environment: selected,
            })
          }
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
