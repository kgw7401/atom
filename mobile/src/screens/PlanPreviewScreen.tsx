import React from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { PlanResponse } from '../api/session';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';
import SectionLabel from '../components/SectionLabel';
import PrimaryButton from '../components/PrimaryButton';

const TEMPO_LABEL: Record<string, string> = { slow: '느림', medium: '보통', fast: '빠름' };
const TEMPO_COLOR: Record<string, string> = { slow: COLORS.TEXT_3, medium: COLORS.GOLD, fast: COLORS.RED };

type Props = NativeStackScreenProps<any, 'PlanPreview'>;

export default function PlanPreviewScreen({ route, navigation }: Props) {
  const result: PlanResponse = (route.params as any)?.result;
  const { plan, id: planId, round_duration_sec, rest_sec } = result;

  const totalMin = Math.round((plan.rounds.length * round_duration_sec + (plan.rounds.length - 1) * rest_sec) / 60);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.meta}>
          {plan.rounds.length}라운드 · 약 {totalMin}분
        </Text>

        <View style={styles.divider} />

        {plan.rounds.map((rnd) => (
          <View key={rnd.round} style={styles.roundSection}>
            <View style={styles.roundHeader}>
              <Text style={styles.roundTitle}>ROUND {rnd.round}</Text>
              <Text style={styles.roundDuration}>{round_duration_sec}초</Text>
            </View>
            {rnd.segments.slice(0, 8).map((seg, i) => (
              <View key={i} style={styles.segRow}>
                <View style={[styles.tempoBadge, { borderColor: (TEMPO_COLOR[seg.tempo] ?? COLORS.TEXT_3) + '80' }]}>
                  <Text style={[styles.tempoBadgeText, { color: TEMPO_COLOR[seg.tempo] ?? COLORS.TEXT_3 }]}>
                    {TEMPO_LABEL[seg.tempo] ?? seg.tempo}
                  </Text>
                </View>
                <Text style={styles.segText} numberOfLines={1}>{seg.text}</Text>
                <Text style={styles.segDur}>{seg.pause_sec ?? seg.duration ?? '–'}s</Text>
              </View>
            ))}
            {rnd.segments.length > 8 && (
              <Text style={styles.more}>+{rnd.segments.length - 8}개 더</Text>
            )}
          </View>
        ))}

        <View style={{ height: 100 }} />
      </ScrollView>

      <View style={styles.ctaContainer}>
        <View style={styles.ctaGradient} />
        <View style={styles.ctaInner}>
          <PrimaryButton
            label="세션 시작 →"
            onPress={() => navigation.navigate('ActiveSession', { planId, plan, round_duration_sec, rest_sec })}
          />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.BG },
  content: { padding: SPACING.PADDING_SCREEN, paddingBottom: 120 },

  meta: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize, marginBottom: SPACING.GAP_SECTION },

  divider: { height: 1, backgroundColor: COLORS.BORDER, marginBottom: SPACING.GAP_SECTION },

  roundSection: { marginBottom: 20 },
  roundHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  roundTitle: {
    color: COLORS.RED,
    fontSize: TYPOGRAPHY.SECTION_LABEL.fontSize,
    fontWeight: TYPOGRAPHY.SECTION_LABEL.fontWeight,
    letterSpacing: TYPOGRAPHY.SECTION_LABEL.letterSpacing,
    textTransform: 'uppercase',
  },
  roundDuration: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize },

  segRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 3, gap: 8 },
  tempoBadge: {
    backgroundColor: '#1a1a1a',
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    width: 40,
    alignItems: 'center',
  },
  tempoBadgeText: { fontSize: 10, fontWeight: '600' },
  segText: { color: COLORS.TEXT_1, fontSize: 15, flex: 1 },
  segDur: { color: COLORS.TEXT_GHOST, fontSize: 12 },
  more: { color: COLORS.TEXT_3, fontSize: TYPOGRAPHY.META.fontSize, marginTop: 4 },

  ctaContainer: { position: 'absolute', bottom: 0, left: 0, right: 0 },
  ctaGradient: { height: 40, backgroundColor: COLORS.BG, opacity: 0.9 },
  ctaInner: {
    backgroundColor: COLORS.BG,
    paddingHorizontal: SPACING.PADDING_SCREEN,
    paddingBottom: 32,
  },
});
