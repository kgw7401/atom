import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';

type Props = {
  value: string;
  label: string;
};

export default function StatCard({ value, label }: Props) {
  return (
    <View style={styles.card}>
      <Text style={styles.value}>{value}</Text>
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: COLORS.SURFACE,
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.BORDER,
  },
  value: {
    color: COLORS.TEXT_1,
    fontSize: TYPOGRAPHY.STAT_VALUE.fontSize,
    fontWeight: TYPOGRAPHY.STAT_VALUE.fontWeight,
  },
  label: {
    color: COLORS.TEXT_3,
    fontSize: 11,
    marginTop: 4,
  },
});
