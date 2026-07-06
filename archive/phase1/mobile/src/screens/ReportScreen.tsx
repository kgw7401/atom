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
import type { SessionReport } from '../types';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Report'>;

export default function ReportScreen({ route, navigation }: Props) {
  const { sessionId } = route.params;
  const [report, setReport] = useState<SessionReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchReport();
  }, []);

  const fetchReport = async () => {
    try {
      const data = await ApiClient.getSessionReport(sessionId);
      setReport(data);
    } catch (error) {
      console.error('Failed to fetch report:', error);
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

  if (!report) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorText}>Failed to load report</Text>
      </View>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return '#2ecc71';
      case 'partial':
        return '#f39c12';
      case 'missed':
        return '#e74c3c';
      default:
        return '#888';
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Overall Score */}
      <View style={styles.scoreContainer}>
        <Text style={styles.scoreLabel}>Overall Score</Text>
        <Text style={styles.scoreValue}>{report.overall_score}</Text>
        <Text style={styles.scoreMax}>/ 100</Text>
      </View>

      {/* Summary */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Summary</Text>
        <View style={styles.summaryGrid}>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>{report.summary.total_instructions}</Text>
            <Text style={styles.summaryLabel}>Total</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={[styles.summaryValue, { color: '#2ecc71' }]}>
              {report.summary.success}
            </Text>
            <Text style={styles.summaryLabel}>Success</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={[styles.summaryValue, { color: '#e74c3c' }]}>
              {report.summary.missed}
            </Text>
            <Text style={styles.summaryLabel}>Missed</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>
              {report.summary.avg_reaction_time.toFixed(2)}s
            </Text>
            <Text style={styles.summaryLabel}>Avg Reaction</Text>
          </View>
        </View>
      </View>

      {/* Instruction Results */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Detailed Results</Text>
        {report.instructions.map((instruction, index) => (
          <View key={index} style={styles.instructionCard}>
            <View style={styles.instructionHeader}>
              <Text style={styles.instructionCommand}>{instruction.command}</Text>
              <View
                style={[
                  styles.statusBadge,
                  { backgroundColor: getStatusColor(instruction.status) },
                ]}
              >
                <Text style={styles.statusText}>{instruction.status}</Text>
              </View>
            </View>
            <View style={styles.instructionDetails}>
              <Text style={styles.detailText}>Score: {instruction.score}/100</Text>
              {instruction.reaction_time !== null && (
                <Text style={styles.detailText}>
                  Reaction: {instruction.reaction_time.toFixed(2)}s
                </Text>
              )}
            </View>
            {instruction.feedback && (
              <Text style={styles.feedbackText}>{instruction.feedback}</Text>
            )}
          </View>
        ))}
      </View>

      {/* Coaching */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Coaching Feedback</Text>

        {report.coaching.strengths.length > 0 && (
          <View style={styles.coachingBox}>
            <Text style={styles.coachingTitle}>💪 Strengths</Text>
            {report.coaching.strengths.map((strength, index) => (
              <Text key={index} style={styles.coachingText}>
                • {strength}
              </Text>
            ))}
          </View>
        )}

        {report.coaching.weaknesses.length > 0 && (
          <View style={styles.coachingBox}>
            <Text style={styles.coachingTitle}>🎯 Areas to Improve</Text>
            {report.coaching.weaknesses.map((weakness, index) => (
              <Text key={index} style={styles.coachingText}>
                • {weakness}
              </Text>
            ))}
          </View>
        )}

        {report.coaching.next_session && (
          <View style={styles.coachingBox}>
            <Text style={styles.coachingTitle}>📋 Next Session</Text>
            <Text style={styles.coachingText}>{report.coaching.next_session}</Text>
          </View>
        )}
      </View>

      {/* Actions */}
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
  },
  errorText: {
    color: '#e74c3c',
    fontSize: 18,
  },
  scoreContainer: {
    alignItems: 'center',
    paddingVertical: 40,
    backgroundColor: '#111',
  },
  scoreLabel: {
    color: '#888',
    fontSize: 16,
    marginBottom: 10,
  },
  scoreValue: {
    color: '#e74c3c',
    fontSize: 72,
    fontWeight: 'bold',
  },
  scoreMax: {
    color: '#666',
    fontSize: 24,
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
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  summaryItem: {
    width: '48%',
    backgroundColor: '#111',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 10,
  },
  summaryValue: {
    color: '#fff',
    fontSize: 28,
    fontWeight: 'bold',
  },
  summaryLabel: {
    color: '#888',
    fontSize: 14,
    marginTop: 5,
  },
  instructionCard: {
    backgroundColor: '#111',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  instructionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  instructionCommand: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  instructionDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  detailText: {
    color: '#888',
    fontSize: 14,
  },
  feedbackText: {
    color: '#aaa',
    fontSize: 14,
    fontStyle: 'italic',
  },
  coachingBox: {
    backgroundColor: '#111',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  coachingTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  coachingText: {
    color: '#aaa',
    fontSize: 14,
    marginBottom: 5,
  },
  button: {
    backgroundColor: '#e74c3c',
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
