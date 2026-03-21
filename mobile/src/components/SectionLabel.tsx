import React from 'react';
import { Text, StyleSheet } from 'react-native';
import { COLORS, TYPOGRAPHY } from '../theme';

type Props = {
  text: string;
  color?: string;
};

export default function SectionLabel({ text, color }: Props) {
  return (
    <Text style={[styles.label, color ? { color } : undefined]}>
      {text}
    </Text>
  );
}

const styles = StyleSheet.create({
  label: {
    color: COLORS.TEXT_2,
    fontSize: TYPOGRAPHY.SECTION_LABEL.fontSize,
    fontWeight: TYPOGRAPHY.SECTION_LABEL.fontWeight,
    letterSpacing: TYPOGRAPHY.SECTION_LABEL.letterSpacing,
    textTransform: 'uppercase',
    marginBottom: 12,
  },
});
