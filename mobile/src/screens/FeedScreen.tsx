import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { fetchSessions, SessionSummary } from '../api/session';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - d.getTime()) / (1000 * 60 * 60 * 24));
  if (diffDays === 0) return '오늘';
  if (diffDays === 1) return '어제';
  if (diffDays < 7) return `${diffDays}일 전`;
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  return m >= 1 ? `${m}분` : `${Math.round(sec)}초`;
}

function SessionCard({ item }: { item: SessionSummary }) {
  const isCompleted = item.status === 'completed';
  return (
    <View style={[styles.card, !isCompleted && styles.cardAbandoned]}>
      <View style={styles.cardTop}>
        <Text style={styles.cardFocus} numberOfLines={1}>복싱 훈련</Text>
        <Text style={styles.cardDate}>{formatDate(item.started_at)}</Text>
      </View>
      <View style={styles.cardMeta}>
        <Text style={styles.metaItem}>
          {item.rounds_completed}/{item.rounds_total}R
        </Text>
        <Text style={styles.metaDot}>·</Text>
        <Text style={styles.metaItem}>{formatDuration(item.total_duration_sec)}</Text>
        <Text style={styles.metaDot}>·</Text>
        <Text style={styles.metaItem}>{item.segments_delivered}개</Text>
        <Text style={styles.metaDot}>·</Text>
        <Text style={[styles.statusBadge, isCompleted ? styles.statusDone : styles.statusAbandoned]}>
          {isCompleted ? '완료' : '중단'}
        </Text>
      </View>
    </View>
  );
}

export default function FeedScreen() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      fetchSessions(30)
        .then(setSessions)
        .catch(() => {})
        .finally(() => setLoading(false));
    }, []),
  );

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={COLORS.RED} />
      </View>
    );
  }

  if (sessions.length === 0) {
    return (
      <View style={styles.center}>
        <Text style={styles.empty}>아직 훈련 기록이 없습니다.</Text>
        <Text style={styles.emptySub}>첫 세션을 완료해보세요!</Text>
      </View>
    );
  }

  const completed = sessions.filter((s) => s.status === 'completed');
  const totalSegments = completed.reduce((sum, s) => sum + s.segments_delivered, 0);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Summary */}
      <View style={styles.summaryRow}>
        <View style={styles.summaryItem}>
          <Text style={styles.summaryValue}>{sessions.length}</Text>
          <Text style={styles.summaryLabel}>총 세션</Text>
        </View>
        <View style={styles.summaryDivider} />
        <View style={styles.summaryItem}>
          <Text style={styles.summaryValue}>{completed.length}</Text>
          <Text style={styles.summaryLabel}>완료</Text>
        </View>
        <View style={styles.summaryDivider} />
        <View style={styles.summaryItem}>
          <Text style={styles.summaryValue}>{totalSegments}</Text>
          <Text style={styles.summaryLabel}>총 구간</Text>
        </View>
      </View>

      {/* Session list */}
      {sessions.map((s) => <SessionCard key={s.id} item={s} />)}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.BG },
  content: { padding: SPACING.PADDING_SCREEN, paddingBottom: 48 },
  center: { flex: 1, backgroundColor: COLORS.BG, alignItems: 'center', justifyContent: 'center' },
  empty: { color: COLORS.TEXT_2, fontSize: 16 },
  emptySub: { color: COLORS.TEXT_3, fontSize: 13, marginTop: 8 },

  summaryRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.SURFACE,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    borderRadius: 14,
    padding: 20,
    marginBottom: 20,
  },
  summaryItem: { flex: 1, alignItems: 'center' },
  summaryValue: { color: COLORS.TEXT_1, fontSize: 22, fontWeight: '700' },
  summaryLabel: { color: COLORS.TEXT_3, fontSize: 12, marginTop: 4 },
  summaryDivider: { width: 1, height: 36, backgroundColor: COLORS.BORDER },

  card: {
    backgroundColor: COLORS.SURFACE,
    borderWidth: 1,
    borderColor: '#2a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  cardAbandoned: { borderColor: COLORS.BORDER },
  cardTop: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  cardFocus: { color: COLORS.TEXT_1, fontSize: 15, fontWeight: '600', flex: 1, marginRight: 8 },
  cardDate: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize },
  cardMeta: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  metaItem: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize },
  metaDot: { color: COLORS.TEXT_GHOST, fontSize: 12 },
  statusBadge: { fontSize: 12, fontWeight: '600' },
  statusDone: { color: COLORS.RED },
  statusAbandoned: { color: COLORS.TEXT_GHOST },
});
