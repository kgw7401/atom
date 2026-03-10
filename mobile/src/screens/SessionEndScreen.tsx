import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

type Props = NativeStackScreenProps<any, 'SessionEnd'>;

export default function SessionEndScreen({ route, navigation }: Props) {
  const { status, rounds, combos, duration } = route.params ?? {};
  const isCompleted = status === 'completed';

  return (
    <View style={styles.container}>
      <Text style={styles.icon}>{isCompleted ? '✓' : '!'}</Text>
      <Text style={styles.title}>{isCompleted ? '완료!' : '중단됨'}</Text>

      <View style={styles.stats}>
        <StatRow label="라운드" value={`${rounds ?? 0}`} />
        <StatRow label="콤보" value={`${combos ?? 0}`} />
        <StatRow label="시간" value={`${Math.round(duration ?? 0)}s`} />
      </View>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Home')}
      >
        <Text style={styles.buttonText}>홈으로</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.secondaryButton}
        onPress={() => navigation.navigate('SessionSetup')}
      >
        <Text style={styles.secondaryButtonText}>다시 시작</Text>
      </TouchableOpacity>
    </View>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.statRow}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1, backgroundColor: '#0a0a0a',
    alignItems: 'center', justifyContent: 'center', padding: 32,
  },
  icon: { fontSize: 64, marginBottom: 16 },
  title: { color: '#fff', fontSize: 36, fontWeight: '700', marginBottom: 40 },
  stats: { width: '100%', backgroundColor: '#1a1a1a', borderRadius: 12, padding: 20, marginBottom: 40 },
  statRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10 },
  statLabel: { color: '#888', fontSize: 16 },
  statValue: { color: '#fff', fontSize: 16, fontWeight: '600' },
  button: {
    backgroundColor: '#e63946', borderRadius: 10,
    padding: 18, width: '100%', alignItems: 'center', marginBottom: 12,
  },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '700' },
  secondaryButton: {
    borderWidth: 1, borderColor: '#333', borderRadius: 10,
    padding: 16, width: '100%', alignItems: 'center',
  },
  secondaryButtonText: { color: '#888', fontSize: 16 },
});
