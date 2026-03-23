import React, { useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, TYPOGRAPHY, SPACING } from '../theme';
import { fetchToday, TodayData } from '../api/session';
import { updateProfile } from '../api/profile';

type Props = { navigation: NativeStackNavigationProp<any> };

const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
const LEVEL_LABELS: Record<string, string> = {
  beginner: '입문',
  novice: '입문',
  intermediate: '중급',
  advanced: '고급',
};

export default function HomeScreen({ navigation }: Props) {
  const [today, setToday] = useState<TodayData | null>(null);
  const [loading, setLoading] = useState(true);

  const handleLevelChange = () => {
    const raw = today?.level ?? 'beginner';
    const current = raw === 'novice' ? 'beginner' : raw;
    const options = LEVELS.filter((l) => l !== current);
    Alert.alert(
      '레벨 변경',
      `현재: ${LEVEL_LABELS[current]}`,
      [
        ...options.map((lvl) => ({
          text: LEVEL_LABELS[lvl],
          onPress: async () => {
            try {
              await updateProfile({ experience_level: lvl });
              const data = await fetchToday();
              setToday(data);
            } catch (e) {
              console.error('Level change failed:', e);
            }
          },
        })),
        { text: '취소', style: 'cancel' as const },
      ],
    );
  };

  useFocusEffect(
    useCallback(() => {
      let cancelled = false;
      setLoading(true);
      fetchToday()
        .then((data) => { if (!cancelled) setToday(data); })
        .catch((e) => console.warn('fetchToday failed:', e))
        .finally(() => { if (!cancelled) setLoading(false); });
      return () => { cancelled = true; };
    }, [])
  );

  if (loading) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator color={COLORS.RED} size="large" />
      </View>
    );
  }

  const streak = today?.streak ?? 0;

  return (
    <View style={styles.container}>
      {/* Logo */}
      <Text style={styles.logo}>ATOM</Text>

      {/* Center: Punching Bag Button */}
      <View style={styles.center}>
        <TouchableOpacity
          style={styles.bagButton}
          onPress={() => navigation.navigate('SessionPicker', { today })}
          activeOpacity={0.8}
        >
          {/* Hanger */}
          <View style={styles.bagHanger} />
          {/* Chain */}
          <View style={styles.bagChain} />
          {/* Bag body */}
          <View style={styles.bagBody}>
            <Text style={styles.bagText}>TAP</Text>
          </View>
        </TouchableOpacity>
      </View>

      {/* Bottom info */}
      <View style={styles.bottomRow}>
        <TouchableOpacity style={styles.levelBadge} onPress={handleLevelChange}>
          <Text style={styles.levelText}>
            {LEVEL_LABELS[today?.level ?? 'beginner']}
          </Text>
        </TouchableOpacity>
        {streak > 0 && (
          <View style={styles.streakBadge}>
            <Text style={styles.streakText}>{streak}일 연속</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const BAG_WIDTH = 100;
const BAG_HEIGHT = 140;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    paddingHorizontal: SPACING.PADDING_SCREEN,
    paddingTop: 80,
    paddingBottom: 50,
    alignItems: 'center',
  },
  logo: {
    ...TYPOGRAPHY.APP_TITLE,
    color: COLORS.RED,
    textAlign: 'center',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bagButton: {
    alignItems: 'center',
  },
  bagHanger: {
    width: 40,
    height: 3,
    backgroundColor: COLORS.TEXT_3,
    borderRadius: 1.5,
  },
  bagChain: {
    width: 2,
    height: 20,
    backgroundColor: COLORS.TEXT_3,
  },
  bagBody: {
    width: BAG_WIDTH,
    height: BAG_HEIGHT,
    borderRadius: BAG_WIDTH / 2,
    backgroundColor: COLORS.RED,
    alignItems: 'center',
    justifyContent: 'center',
    // Punching bag shape: wider at top, narrower at bottom
    borderTopLeftRadius: BAG_WIDTH / 2.2,
    borderTopRightRadius: BAG_WIDTH / 2.2,
    borderBottomLeftRadius: BAG_WIDTH / 2.8,
    borderBottomRightRadius: BAG_WIDTH / 2.8,
  },
  bagText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
    letterSpacing: 3,
    opacity: 0.9,
  },
  bottomRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  levelBadge: {
    backgroundColor: COLORS.SURFACE,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: SPACING.RADIUS_BADGE,
    borderWidth: 1,
    borderColor: COLORS.TEXT_3,
  },
  levelText: {
    color: COLORS.TEXT_2,
    ...TYPOGRAPHY.META,
    fontWeight: '700',
  },
  streakBadge: {
    backgroundColor: COLORS.SURFACE,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: SPACING.RADIUS_BADGE,
    borderWidth: 1,
    borderColor: COLORS.ORANGE,
  },
  streakText: {
    color: COLORS.ORANGE,
    ...TYPOGRAPHY.META,
    fontWeight: '700',
  },
});
