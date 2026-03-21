import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { COLORS } from '../theme';

type Props = { navigation: NativeStackNavigationProp<any> };

export default function HomeScreen({ navigation }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.top}>
        <Text style={styles.logo}>ATOM</Text>
        <Text style={styles.sub}>AI 복싱 코치</Text>
      </View>

      <TouchableOpacity
        style={styles.startBtn}
        onPress={() => navigation.navigate('SessionSetup')}
        activeOpacity={0.85}
      >
        <Text style={styles.startBtnText}>훈련 시작</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    paddingHorizontal: 32,
    justifyContent: 'space-between',
    paddingTop: 120,
    paddingBottom: 80,
  },
  top: {
    alignItems: 'center',
  },
  logo: {
    color: COLORS.RED,
    fontSize: 52,
    fontWeight: '900',
    letterSpacing: 8,
  },
  sub: {
    color: COLORS.TEXT_3,
    fontSize: 15,
    marginTop: 8,
    letterSpacing: 1,
  },
  startBtn: {
    backgroundColor: COLORS.RED,
    borderRadius: 16,
    paddingVertical: 20,
    alignItems: 'center',
  },
  startBtnText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '700',
    letterSpacing: 1,
  },
});
