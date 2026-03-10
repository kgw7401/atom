import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { fetchSessions, SessionSummary } from '../api/session';

type Props = { navigation: NativeStackNavigationProp<any> };

export default function HistoryScreen({ navigation }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSessions()
      .then(setSessions)
      .catch(() => Alert.alert('Error', 'Cannot connect to server.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#e63946" />
      </View>
    );
  }

  return (
    <FlatList
      style={styles.container}
      data={sessions}
      keyExtractor={(item) => item.id}
      ListEmptyComponent={
        <Text style={styles.empty}>No sessions yet. Start training!</Text>
      }
      renderItem={({ item }) => (
        <TouchableOpacity
          style={styles.card}
          onPress={() => navigation.navigate('SessionDetail', { sessionId: item.id })}
        >
          <View style={styles.cardRow}>
            <Text style={styles.template}>{item.template_name}</Text>
            <Text style={[styles.status, item.status === 'completed' ? styles.statusDone : styles.statusAborted]}>
              {item.status}
            </Text>
          </View>
          <Text style={styles.date}>{new Date(item.started_at).toLocaleString('ko-KR')}</Text>
          <Text style={styles.stats}>
            {item.rounds_completed}/{item.rounds_total} rounds · {item.combos_delivered} combos · {Math.round(item.total_duration_sec)}s
          </Text>
        </TouchableOpacity>
      )}
    />
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a', padding: 16 },
  center: { flex: 1, backgroundColor: '#0a0a0a', alignItems: 'center', justifyContent: 'center' },
  empty: { color: '#555', textAlign: 'center', marginTop: 60, fontSize: 16 },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#222',
  },
  cardRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  template: { color: '#fff', fontSize: 16, fontWeight: '600' },
  status: { fontSize: 12, fontWeight: '600', textTransform: 'uppercase' },
  statusDone: { color: '#4caf50' },
  statusAborted: { color: '#ff9800' },
  date: { color: '#666', fontSize: 12, marginBottom: 4 },
  stats: { color: '#888', fontSize: 13 },
});
