import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
  withSpring,
  withDelay,
  Easing,
  interpolate,
} from 'react-native-reanimated';
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

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

export default function HomeScreen({ navigation }: Props) {
  const [today, setToday] = useState<TodayData | null>(null);
  const [loading, setLoading] = useState(true);

  // ── Animations ──────────────────────────────────────────────────
  const logoOpacity = useSharedValue(0);
  const logoTranslateY = useSharedValue(-20);
  const bagSwing = useSharedValue(0);
  const bagScale = useSharedValue(0);
  const glowOpacity = useSharedValue(0.3);
  const bottomOpacity = useSharedValue(0);

  useEffect(() => {
    // Logo fade-in from top
    logoOpacity.value = withDelay(200, withTiming(1, { duration: 600 }));
    logoTranslateY.value = withDelay(200, withTiming(0, { duration: 600, easing: Easing.out(Easing.cubic) }));

    // Bag entrance: scale up with spring
    bagScale.value = withDelay(500, withSpring(1, { damping: 12, stiffness: 100 }));

    // Idle sway: gentle pendulum (starts after entrance)
    bagSwing.value = withDelay(1200,
      withRepeat(
        withSequence(
          withTiming(1, { duration: 2400, easing: Easing.inOut(Easing.sin) }),
          withTiming(-1, { duration: 2400, easing: Easing.inOut(Easing.sin) }),
        ),
        -1, // infinite
        true,
      )
    );

    // Glow pulse
    glowOpacity.value = withDelay(800,
      withRepeat(
        withSequence(
          withTiming(0.6, { duration: 2000, easing: Easing.inOut(Easing.sin) }),
          withTiming(0.2, { duration: 2000, easing: Easing.inOut(Easing.sin) }),
        ),
        -1,
        true,
      )
    );

    // Bottom badges fade in
    bottomOpacity.value = withDelay(900, withTiming(1, { duration: 500 }));
  }, []);

  const logoStyle = useAnimatedStyle(() => ({
    opacity: logoOpacity.value,
    transform: [{ translateY: logoTranslateY.value }],
  }));

  const bagContainerStyle = useAnimatedStyle(() => {
    const rotate = interpolate(bagSwing.value, [-1, 1], [-1.5, 1.5]);
    return {
      transform: [
        { scale: bagScale.value },
        { rotate: `${rotate}deg` },
      ],
    };
  });

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  const bottomStyle = useAnimatedStyle(() => ({
    opacity: bottomOpacity.value,
  }));

  // ── Tap handler with press animation ────────────────────────────
  const handlePress = () => {
    // Quick punch reaction: squish then spring back
    bagScale.value = withSequence(
      withTiming(0.92, { duration: 80 }),
      withSpring(1, { damping: 8, stiffness: 200 }),
    );
    // Navigate after brief delay for visual feedback
    setTimeout(() => {
      navigation.navigate('SessionPicker', { today });
    }, 150);
  };

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
      <Animated.Text style={[styles.logo, logoStyle]}>ATOM</Animated.Text>

      {/* Center: Punching Bag */}
      <View style={styles.center}>
        {/* Glow behind bag */}
        <Animated.View style={[styles.glow, glowStyle]} />

        <Animated.View style={[styles.bagAssembly, bagContainerStyle]}>
          <TouchableOpacity
            style={styles.bagButton}
            onPress={handlePress}
            activeOpacity={1}
          >
            {/* Mount */}
            <View style={styles.mountPlate} />
            <View style={styles.mountBolt} />
            {/* Chains */}
            <View style={styles.chainGroup}>
              <View style={[styles.chain, styles.chainLeft]} />
              <View style={[styles.chain, styles.chainCenter]} />
              <View style={[styles.chain, styles.chainRight]} />
            </View>
            {/* Swivel */}
            <View style={styles.swivel} />
            {/* Bag body */}
            <View style={styles.bagBody}>
              <View style={styles.bagCap} />
              <View style={styles.bagHighlight} />
              <View style={styles.bagSeam} />
            </View>
            {/* Bottom cap */}
            <View style={styles.bagBottom} />
          </TouchableOpacity>
        </Animated.View>
      </View>

      {/* Bottom info */}
      <Animated.View style={[styles.bottomRow, bottomStyle]}>
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
      </Animated.View>
    </View>
  );
}

const BAG_WIDTH = 90;
const BAG_HEIGHT = 200;
const BAG_COLOR = '#8B1A1A';
const BAG_HIGHLIGHT = '#A52222';

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
  glow: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: COLORS.RED,
  },
  bagAssembly: {
    alignItems: 'center',
    // Pivot from top (mount point)
    transformOrigin: 'top center',
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
