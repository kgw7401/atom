import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { ApiClient } from '../services/api';
import type { TwinResponse } from '../types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Twin'>;

export default function TwinScreen({ route, navigation }: Props) {
  const { userId } = route.params;
  const [twin, setTwin] = useState<TwinResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTwin();
  }, []);

  const fetchTwin = async () => {
    try {
      const data = await ApiClient.getTwin(userId);
      setTwin(data);
    } catch (error) {
      console.error('Failed to fetch twin:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#e74c3c" />
      </View>
    );
  }

  if (!twin) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorText}>No data available yet</Text>
        <Text style={styles.subtitleText}>Complete a session to see your stats</Text>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('Home')}
        >
          <Text style={styles.buttonText}>Back to Home</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return '📈';
      case 'declining':
        return '📉';
      default:
        return '➡️';
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Digital Twin</Text>
        <Text style={styles.subtitle}>{twin.total_sessions} Sessions Completed</Text>
      </View>

      {/* Per-Action Stats */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Performance by Action</Text>
        {Object.entries(twin.per_action_stats).map(([action, stats]) => (
          <View key={action} style={styles.statCard}>
            <View style={styles.statHeader}>
              <Text style={styles.statAction}>{action.replace('_', ' ')}</Text>
              <Text style={styles.trendIcon}>{getTrendIcon(stats.trend)}</Text>
            </View>
            <View style={styles.statRow}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {(stats.accuracy * 100).toFixed(0)}%
                </Text>
                <Text style={styles.statLabel}>Accuracy</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {stats.avg_reaction.toFixed(2)}s
                </Text>
                <Text style={styles.statLabel}>Avg Reaction</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{stats.total_attempts}</Text>
                <Text style={styles.statLabel}>Attempts</Text>
              </View>
            </View>
          </View>
        ))}
      </View>

      {/* Weaknesses */}
      {twin.weaknesses.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Areas to Improve</Text>
          {twin.weaknesses.map((weakness, index) => (
            <View key={index} style={styles.weaknessCard}>
              <Text style={styles.weaknessAction}>
                {weakness.action.replace('_', ' ')}
              </Text>
              <Text style={styles.weaknessText}>
                {weakness.metric === 'accuracy' && (
                  <>
                    Accuracy {(weakness.value * 100).toFixed(0)}% (target:{' '}
                    {(weakness.threshold * 100).toFixed(0)}%)
                  </>
                )}
                {weakness.metric === 'reaction_time' && (
                  <>
                    Reaction time {weakness.value.toFixed(2)}s (target:{' '}
                    {'<'}{weakness.threshold.toFixed(2)}s)
                  </>
                )}
              </Text>
            </View>
          ))}
        </View>
      )}

      {/* Growth Curves */}
      {twin.growth_curves.scores.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Score Progression</Text>
          <View style={styles.chartContainer}>
            {twin.growth_curves.scores.map((score, index) => (
              <View key={index} style={styles.chartBar}>
                <View
                  style={[
                    styles.chartBarFill,
                    { height: `${score}%` },
                  ]}
                />
                <Text style={styles.chartLabel}>{index + 1}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Home')}
      >
        <Text style={styles.buttonText}>Back to Home</Text>
      </TouchableOpacity>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  subtitleText: {
    color: '#888',
    fontSize: 16,
    marginBottom: 30,
  },
  header: {
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#111',
    alignItems: 'center',
  },
  title: {
    color: '#fff',
    fontSize: 32,
    fontWeight: 'bold',
  },
  subtitle: {
    color: '#888',
    fontSize: 16,
    marginTop: 8,
  },
  section: {
    padding: 20,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  statCard: {
    backgroundColor: '#111',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  statHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statAction: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  trendIcon: {
    fontSize: 20,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    color: '#e74c3c',
    fontSize: 20,
    fontWeight: 'bold',
  },
  statLabel: {
    color: '#888',
    fontSize: 12,
    marginTop: 4,
  },
  weaknessCard: {
    backgroundColor: '#1a0a0a',
    borderLeftWidth: 4,
    borderLeftColor: '#e74c3c',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
  },
  weaknessAction: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
    textTransform: 'capitalize',
  },
  weaknessText: {
    color: '#aaa',
    fontSize: 14,
  },
  chartContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-around',
    height: 150,
    backgroundColor: '#111',
    padding: 20,
    borderRadius: 10,
  },
  chartBar: {
    flex: 1,
    alignItems: 'center',
    marginHorizontal: 2,
  },
  chartBarFill: {
    width: '80%',
    backgroundColor: '#e74c3c',
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
    minHeight: 10,
  },
  chartLabel: {
    color: '#888',
    fontSize: 12,
    marginTop: 8,
  },
  button: {
    backgroundColor: '#333',
    padding: 18,
    borderRadius: 10,
    marginHorizontal: 20,
    marginTop: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
