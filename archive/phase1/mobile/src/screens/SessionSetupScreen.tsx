import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import { ApiClient } from '../services/api';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'SessionSetup'>;

export default function SessionSetupScreen({ route, navigation }: Props) {
  const { userId } = route.params;
  const [level, setLevel] = useState(1);
  const [duration, setDuration] = useState(60);
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    setLoading(true);
    try {
      // Generate script
      const script = await ApiClient.generateScript(level, duration);

      // Create session
      const startedAt = new Date().toISOString();
      const session = await ApiClient.createSession(
        userId,
        script.script_id,
        startedAt
      );

      // Navigate to drill session
      navigation.navigate('DrillSession', {
        sessionId: session.session_id,
        script,
      });
    } catch (error) {
      console.error('Failed to setup session:', error);
      Alert.alert('Error', 'Failed to create session. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Setup Session</Text>

      {/* Level Selection */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Select Level</Text>
        <View style={styles.levelContainer}>
          {[1, 2, 3].map((l) => (
            <TouchableOpacity
              key={l}
              style={[styles.levelButton, level === l && styles.levelButtonActive]}
              onPress={() => setLevel(l)}
            >
              <Text style={[styles.levelText, level === l && styles.levelTextActive]}>
                Level {l}
              </Text>
              <Text style={styles.levelDesc}>
                {l === 1 ? 'Beginner' : l === 2 ? 'Intermediate' : 'Advanced'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Duration Selection */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Select Duration</Text>
        <View style={styles.durationContainer}>
          {[30, 60, 120, 180].map((d) => (
            <TouchableOpacity
              key={d}
              style={[
                styles.durationButton,
                duration === d && styles.durationButtonActive,
              ]}
              onPress={() => setDuration(d)}
            >
              <Text
                style={[
                  styles.durationText,
                  duration === d && styles.durationTextActive,
                ]}
              >
                {d < 60 ? `${d}s` : `${d / 60}m`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Start Button */}
      <TouchableOpacity
        style={[styles.startButton, loading && styles.startButtonDisabled]}
        onPress={handleStart}
        disabled={loading}
      >
        <Text style={styles.startButtonText}>
          {loading ? 'Loading...' : 'Start Session'}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.backButton}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.backButtonText}>Back</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 40,
    marginBottom: 40,
  },
  section: {
    marginBottom: 40,
  },
  sectionTitle: {
    fontSize: 18,
    color: '#fff',
    marginBottom: 15,
    fontWeight: '600',
  },
  levelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  levelButton: {
    flex: 1,
    backgroundColor: '#222',
    padding: 20,
    borderRadius: 10,
    marginHorizontal: 5,
    alignItems: 'center',
  },
  levelButtonActive: {
    backgroundColor: '#e74c3c',
  },
  levelText: {
    color: '#888',
    fontSize: 16,
    fontWeight: 'bold',
  },
  levelTextActive: {
    color: '#fff',
  },
  levelDesc: {
    color: '#666',
    fontSize: 12,
    marginTop: 5,
  },
  durationContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  durationButton: {
    backgroundColor: '#222',
    padding: 15,
    borderRadius: 10,
    marginRight: 10,
    marginBottom: 10,
    minWidth: 70,
    alignItems: 'center',
  },
  durationButtonActive: {
    backgroundColor: '#e74c3c',
  },
  durationText: {
    color: '#888',
    fontSize: 16,
    fontWeight: '600',
  },
  durationTextActive: {
    color: '#fff',
  },
  startButton: {
    backgroundColor: '#e74c3c',
    padding: 20,
    borderRadius: 10,
    marginTop: 20,
  },
  startButtonDisabled: {
    backgroundColor: '#666',
  },
  startButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  backButton: {
    padding: 15,
    marginTop: 10,
  },
  backButtonText: {
    color: '#888',
    fontSize: 16,
    textAlign: 'center',
  },
});
