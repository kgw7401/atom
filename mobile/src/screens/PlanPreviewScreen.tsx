import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { NativeStackNavigationProp, NativeStackScreenProps } from '@react-navigation/native-stack';
import { PlanResponse } from '../api/session';

type Props = NativeStackScreenProps<any, 'PlanPreview'>;

export default function PlanPreviewScreen({ route, navigation }: Props) {
  const result: PlanResponse = route.params?.result;
  const { plan, id: planId, llm_model } = result;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.focus}>{plan.focus}</Text>
      <Text style={styles.meta}>
        {plan.rounds.length}라운드 · ~{plan.total_duration_minutes}분 · {llm_model}
      </Text>

      {plan.rounds.map((rnd) => (
        <View key={rnd.round_number} style={styles.roundCard}>
          <Text style={styles.roundTitle}>
            Round {rnd.round_number}  {rnd.duration_seconds}s + Rest {rnd.rest_after_seconds}s
          </Text>
          {rnd.instructions.slice(0, 6).map((instr, i) => (
            <Text key={i} style={styles.combo}>
              {instr.combo_display_name}
              <Text style={styles.comboActions}>  {instr.actions.join(' → ')}</Text>
            </Text>
          ))}
          {rnd.instructions.length > 6 && (
            <Text style={styles.more}>+{rnd.instructions.length - 6} more</Text>
          )}
        </View>
      ))}

      <TouchableOpacity
        style={styles.startButton}
        onPress={() => navigation.navigate('ActiveSession', { planId, plan })}
      >
        <Text style={styles.startButtonText}>세션 시작 →</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a' },
  content: { padding: 24, paddingBottom: 48 },
  focus: { color: '#fff', fontSize: 24, fontWeight: '700', marginBottom: 6 },
  meta: { color: '#888', fontSize: 13, marginBottom: 24 },
  roundCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#222',
  },
  roundTitle: { color: '#e63946', fontSize: 13, fontWeight: '600', marginBottom: 10, textTransform: 'uppercase' },
  combo: { color: '#fff', fontSize: 15, marginBottom: 4 },
  comboActions: { color: '#666', fontSize: 13 },
  more: { color: '#555', fontSize: 13, marginTop: 4 },
  startButton: {
    backgroundColor: '#e63946',
    borderRadius: 10,
    padding: 18,
    alignItems: 'center',
    marginTop: 16,
  },
  startButtonText: { color: '#fff', fontSize: 18, fontWeight: '700' },
});
