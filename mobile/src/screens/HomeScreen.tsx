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
          {/* Ceiling mount bracket */}
          <View style={styles.mountPlate} />
          <View style={styles.mountBolt} />
          {/* Chains — two angled + center */}
          <View style={styles.chainGroup}>
            <View style={[styles.chain, styles.chainLeft]} />
            <View style={[styles.chain, styles.chainCenter]} />
            <View style={[styles.chain, styles.chainRight]} />
          </View>
          {/* Swivel ring */}
          <View style={styles.swivel} />
          {/* Bag body */}
          <View style={styles.bagBody}>
            {/* Top cap (leather collar) */}
            <View style={styles.bagCap} />
            {/* Highlight strip for 3D depth */}
            <View style={styles.bagHighlight} />
            {/* Center seam */}
            <View style={styles.bagSeam} />
          </View>
          {/* Bottom cap */}
          <View style={styles.bagBottom} />
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

const BAG_WIDTH = 90;
const BAG_HEIGHT = 200;
const BAG_COLOR = '#8B1A1A';       // dark oxblood leather
const BAG_HIGHLIGHT = '#A52222';   // lighter leather highlight

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
  mountPlate: {
    width: 48,
    height: 6,
    backgroundColor: '#3a3a3a',
    borderRadius: 2,
  },
  mountBolt: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4a4a4a',
    marginTop: -1,
  },
  chainGroup: {
    width: 40,
    height: 28,
    position: 'relative',
  },
  chain: {
    position: 'absolute',
    width: 2,
    height: 28,
    backgroundColor: '#5a5a5a',
    top: 0,
  },
  chainLeft: {
    left: 4,
    transform: [{ rotate: '8deg' }],
  },
  chainCenter: {
    left: 19,
  },
  chainRight: {
    right: 4,
    transform: [{ rotate: '-8deg' }],
  },
  swivel: {
    width: 14,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#4a4a4a',
    marginBottom: 2,
  },
  bagBody: {
    width: BAG_WIDTH,
    height: BAG_HEIGHT,
    backgroundColor: BAG_COLOR,
    borderTopLeftRadius: BAG_WIDTH / 2.5,
    borderTopRightRadius: BAG_WIDTH / 2.5,
    borderBottomLeftRadius: BAG_WIDTH / 3,
    borderBottomRightRadius: BAG_WIDTH / 3,
    overflow: 'hidden',
  },
  bagCap: {
    width: BAG_WIDTH,
    height: 14,
    backgroundColor: '#6B1414',
    borderBottomWidth: 2,
    borderBottomColor: '#4a1010',
  },
  bagHighlight: {
    position: 'absolute',
    left: BAG_WIDTH * 0.22,
    top: 20,
    width: 12,
    height: BAG_HEIGHT - 50,
    backgroundColor: BAG_HIGHLIGHT,
    borderRadius: 6,
    opacity: 0.4,
  },
  bagSeam: {
    position: 'absolute',
    left: BAG_WIDTH / 2 - 0.5,
    top: 16,
    width: 1,
    height: BAG_HEIGHT - 40,
    backgroundColor: '#5a1515',
    opacity: 0.6,
  },
  bagBottom: {
    width: BAG_WIDTH * 0.7,
    height: 10,
    backgroundColor: '#6B1414',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    alignSelf: 'center',
    marginTop: -2,
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
