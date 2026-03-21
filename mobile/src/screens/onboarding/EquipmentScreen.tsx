import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { COLORS, SPACING, TYPOGRAPHY } from '../../theme';
import SectionLabel from '../../components/SectionLabel';
import PrimaryButton from '../../components/PrimaryButton';

type Props = NativeStackScreenProps<any, 'Equipment'>;

const OPTIONS = [
  { key: 'heavy_bag', label: '샌드백', desc: '무거운 백' },
  { key: 'speed_bag', label: '스피드백', desc: '반응 훈련용' },
  { key: 'mitts', label: '미트', desc: '파트너 미트 훈련' },
  { key: 'gloves', label: '글러브', desc: '복싱 글러브' },
  { key: 'jump_rope', label: '줄넘기', desc: '컨디셔닝용' },
  { key: 'none', label: '없음', desc: '장비 없이 맨손' },
];

export default function EquipmentScreen({ route, navigation }: Props) {
  const params = route.params as Record<string, string>;
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const handleToggle = (key: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setSelected((prev) => {
      const next = new Set(prev);
      if (key === 'none') {
        return new Set(['none']);
      }
      next.delete('none');
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
      <SectionLabel text="보유 장비" />
      <Text style={styles.hint}>여러 개 선택 가능</Text>

      <View style={styles.options}>
        {OPTIONS.map((opt) => {
          const isSelected = selected.has(opt.key);
          return (
            <TouchableOpacity
              key={opt.key}
              style={[styles.option, isSelected ? styles.optionSelected : styles.optionDefault]}
              onPress={() => handleToggle(opt.key)}
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
            navigation.navigate('Goal', {
              ...params,
              equipment: Array.from(selected),
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
    marginBottom: 2,
  },
  optionDesc: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize },
  bottom: { flex: 1, justifyContent: 'flex-end', paddingBottom: 32 },
});
