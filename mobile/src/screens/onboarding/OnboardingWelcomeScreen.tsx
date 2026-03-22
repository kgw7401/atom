import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, TYPOGRAPHY, SPACING } from '../../theme';

export default function OnboardingWelcomeScreen({ navigation }: any) {
  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.logo}>ATOM</Text>
        <Text style={styles.tagline}>매일 10분, 복싱 루틴</Text>
      </View>
      <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Experience')}>
        <Text style={styles.buttonText}>시작하기</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 80,
    paddingHorizontal: SPACING.PADDING_SCREEN,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  logo: {
    ...TYPOGRAPHY.APP_TITLE,
    fontSize: 56,
    color: COLORS.TEXT_1,
    letterSpacing: 12,
  },
  tagline: {
    ...TYPOGRAPHY.BODY,
    color: COLORS.TEXT_2,
    marginTop: 16,
  },
  button: {
    backgroundColor: COLORS.RED,
    width: '85%',
    paddingVertical: 18,
    borderRadius: SPACING.RADIUS_CARD,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    ...TYPOGRAPHY.CARD_TITLE,
    fontSize: 18,
  },
});
