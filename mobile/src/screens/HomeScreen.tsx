import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert, Dimensions } from 'react-native';
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

const { width: SCREEN_W } = Dimensions.get('window');

const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
const LEVEL_LABELS: Record<string, string> = {
  beginner: '입문',
  novice: '입문',
  intermediate: '중급',
  advanced: '고급',
};

// ── Floating particle component ─────────────────────────────────────
function FloatingParticle({ delay, x, size }: { delay: number; x: number; size: number }) {
  const translateY = useSharedValue(0);
  const opacity = useSharedValue(0);

  useEffect(() => {
    opacity.value = withDelay(delay,
      withRepeat(
        withSequence(
          withTiming(0.4, { duration: 2000, easing: Easing.inOut(Easing.sin) }),
          withTiming(0, { duration: 2000, easing: Easing.inOut(Easing.sin) }),
        ), -1, true,
      )
    );
    translateY.value = withDelay(delay,
      withRepeat(
        withTiming(-120, { duration: 4000, easing: Easing.inOut(Easing.sin) }),
        -1, true,
      )
    );
  }, []);

  const style = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));

  return (
    <Animated.View
      style={[{
        position: 'absolute',
        left: x,
        bottom: '30%',
        width: size,
        height: size,
        borderRadius: size / 2,
        backgroundColor: COLORS.RED,
      }, style]}
    />
  );
}

export default function HomeScreen({ navigation }: Props) {
  const [today, setToday] = useState<TodayData | null>(null);
  const [loading, setLoading] = useState(true);

  // ── Animations ──────────────────────────────────────────────────
  const logoOpacity = useSharedValue(0);
  const logoTranslateY = useSharedValue(-30);
  const bagSwing = useSharedValue(0);
  const bagScale = useSharedValue(0);
  const glowOpacity = useSharedValue(0);
  const glowScale = useSharedValue(0.8);
  const bottomOpacity = useSharedValue(0);
  const bottomTranslateY = useSharedValue(20);

  useEffect(() => {
    // Logo
    logoOpacity.value = withDelay(200, withTiming(1, { duration: 800 }));
    logoTranslateY.value = withDelay(200, withTiming(0, { duration: 800, easing: Easing.out(Easing.cubic) }));

    // Bag entrance
    bagScale.value = withDelay(400, withSpring(1, { damping: 14, stiffness: 80 }));

    // Idle sway
    bagSwing.value = withDelay(1400,
      withRepeat(
        withSequence(
          withTiming(1, { duration: 2800, easing: Easing.inOut(Easing.sin) }),
          withTiming(-1, { duration: 2800, easing: Easing.inOut(Easing.sin) }),
        ), -1, true,
      )
    );

    // Glow pulse
    glowOpacity.value = withDelay(600,
      withRepeat(
        withSequence(
          withTiming(0.5, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
          withTiming(0.15, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
        ), -1, true,
      )
    );
    glowScale.value = withDelay(600,
      withRepeat(
        withSequence(
          withTiming(1.1, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
          withTiming(0.9, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
        ), -1, true,
      )
    );

    // Bottom
    bottomOpacity.value = withDelay(1000, withTiming(1, { duration: 600 }));
    bottomTranslateY.value = withDelay(1000, withTiming(0, { duration: 600, easing: Easing.out(Easing.cubic) }));
  }, []);

  const logoStyle = useAnimatedStyle(() => ({
    opacity: logoOpacity.value,
    transform: [{ translateY: logoTranslateY.value }],
  }));

  const bagContainerStyle = useAnimatedStyle(() => {
    const rotate = interpolate(bagSwing.value, [-1, 1], [-1.8, 1.8]);
    return {
      transform: [
        { scale: bagScale.value },
        { rotate: `${rotate}deg` },
      ],
    };
  });

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
    transform: [{ scale: glowScale.value }],
  }));

  const bottomStyle = useAnimatedStyle(() => ({
    opacity: bottomOpacity.value,
    transform: [{ translateY: bottomTranslateY.value }],
  }));

  const handlePress = () => {
    bagScale.value = withSequence(
      withTiming(0.9, { duration: 80 }),
      withSpring(1, { damping: 8, stiffness: 200 }),
    );
    setTimeout(() => navigation.navigate('SessionPicker', { today }), 150);
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
      {/* Ambient particles */}
      <FloatingParticle delay={0} x={SCREEN_W * 0.15} size={3} />
      <FloatingParticle delay={800} x={SCREEN_W * 0.75} size={2} />
      <FloatingParticle delay={1600} x={SCREEN_W * 0.55} size={4} />
      <FloatingParticle delay={2400} x={SCREEN_W * 0.3} size={2} />
      <FloatingParticle delay={3200} x={SCREEN_W * 0.85} size={3} />

      {/* Logo */}
      <Animated.Text style={[styles.logo, logoStyle]}>ATOM</Animated.Text>

      {/* Center: Punching Bag */}
      <View style={styles.center}>
        {/* Outer glow ring */}
        <Animated.View style={[styles.glowOuter, glowStyle]} />
        {/* Inner glow */}
        <Animated.View style={[styles.glowInner, glowStyle]} />

        <Animated.View style={[styles.bagAssembly, bagContainerStyle]}>
          <TouchableOpacity style={styles.bagButton} onPress={handlePress} activeOpacity={1}>
            {/* Mount bracket */}
            <View style={styles.mountPlate}>
              <View style={styles.mountRidge} />
            </View>
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
              {/* Collar with rivets */}
              <View style={styles.bagCollar}>
                <View style={styles.collarStrap} />
                <View style={[styles.rivet, { left: 14 }]} />
                <View style={[styles.rivet, { left: BAG_WIDTH / 2 - 3 }]} />
                <View style={[styles.rivet, { right: 14 }]} />
              </View>
              {/* Panel dividers (4-panel look) */}
              <View style={[styles.panelLine, { left: BAG_WIDTH * 0.28 }]} />
              <View style={[styles.panelLine, { left: BAG_WIDTH * 0.5 }]} />
              <View style={[styles.panelLine, { left: BAG_WIDTH * 0.72 }]} />
              {/* Stitch lines along panels */}
              <View style={[styles.stitchLine, { left: BAG_WIDTH * 0.28 - 3 }]} />
              <View style={[styles.stitchLine, { left: BAG_WIDTH * 0.28 + 2 }]} />
              <View style={[styles.stitchLine, { left: BAG_WIDTH * 0.72 - 3 }]} />
              <View style={[styles.stitchLine, { left: BAG_WIDTH * 0.72 + 2 }]} />
              {/* Highlight strips (leather sheen) */}
              <View style={styles.sheen1} />
              <View style={styles.sheen2} />
              {/* Right shadow for roundness */}
              <View style={styles.bagShadowR} />
              {/* Brand patch */}
              <View style={styles.brandPatch}>
                <Text style={styles.brandText}>ATOM</Text>
              </View>
            </View>

            {/* Bottom cap with D-ring */}
            <View style={styles.bagBottom}>
              <View style={styles.bottomStrap} />
              <View style={[styles.rivetSmall, { left: 10 }]} />
              <View style={[styles.rivetSmall, { right: 10 }]} />
            </View>
            {/* D-ring */}
            <View style={styles.dRing} />
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

const BAG_WIDTH = 120;
const BAG_HEIGHT = 280;
const BAG_COLOR = '#8B1A1A';
const BAG_DARK = '#6B1414';
const BAG_DARKER = '#4a1010';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    paddingHorizontal: SPACING.PADDING_SCREEN,
    paddingTop: 70,
    paddingBottom: 44,
    alignItems: 'center',
  },
  logo: {
    fontSize: 36,
    fontWeight: '900',
    letterSpacing: 10,
    color: COLORS.RED,
    textAlign: 'center',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  // ── Glow ──
  glowOuter: {
    position: 'absolute',
    width: 260,
    height: 260,
    borderRadius: 130,
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: COLORS.RED,
  },
  glowInner: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: COLORS.RED,
    opacity: 0.15,
  },
  // ── Bag assembly ──
  bagAssembly: {
    alignItems: 'center',
    transformOrigin: 'top center',
  },
  bagButton: {
    alignItems: 'center',
  },
  mountPlate: {
    width: 56,
    height: 8,
    backgroundColor: '#3a3a3a',
    borderRadius: 3,
    overflow: 'hidden',
  },
  mountRidge: {
    position: 'absolute',
    top: 3,
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: '#4a4a4a',
  },
  mountBolt: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#4a4a4a',
    borderWidth: 1,
    borderColor: '#5a5a5a',
    marginTop: -2,
  },
  chainGroup: {
    width: 50,
    height: 36,
    position: 'relative',
  },
  chain: {
    position: 'absolute',
    width: 2,
    height: 36,
    backgroundColor: '#5a5a5a',
    top: 0,
  },
  chainLeft: {
    left: 5,
    transform: [{ rotate: '10deg' }],
  },
  chainCenter: {
    left: 24,
  },
  chainRight: {
    right: 5,
    transform: [{ rotate: '-10deg' }],
  },
  swivel: {
    width: 16,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4a4a4a',
    borderWidth: 1,
    borderColor: '#5a5a5a',
    marginBottom: 2,
  },
  // ── Bag body ──
  bagBody: {
    width: BAG_WIDTH,
    height: BAG_HEIGHT,
    backgroundColor: BAG_COLOR,
    borderTopLeftRadius: BAG_WIDTH / 2.3,
    borderTopRightRadius: BAG_WIDTH / 2.3,
    borderBottomLeftRadius: BAG_WIDTH / 2.8,
    borderBottomRightRadius: BAG_WIDTH / 2.8,
    overflow: 'hidden',
    // Subtle border for edge definition
    borderWidth: 0.5,
    borderColor: '#5a1010',
  },
  bagCollar: {
    width: BAG_WIDTH,
    height: 20,
    backgroundColor: BAG_DARK,
    borderBottomWidth: 2,
    borderBottomColor: BAG_DARKER,
    flexDirection: 'row',
    alignItems: 'center',
  },
  collarStrap: {
    position: 'absolute',
    top: 8,
    left: 12,
    right: 12,
    height: 3,
    backgroundColor: BAG_DARKER,
    borderRadius: 1.5,
  },
  rivet: {
    position: 'absolute',
    top: 6,
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#6a6a6a',
    borderWidth: 0.5,
    borderColor: '#888',
  },
  rivetSmall: {
    position: 'absolute',
    top: 4,
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#5a5a5a',
    borderWidth: 0.5,
    borderColor: '#777',
  },
  panelLine: {
    position: 'absolute',
    top: 22,
    width: 1.5,
    height: BAG_HEIGHT - 46,
    backgroundColor: '#6a1515',
    opacity: 0.5,
  },
  stitchLine: {
    position: 'absolute',
    top: 24,
    width: 0.5,
    height: BAG_HEIGHT - 50,
    backgroundColor: '#9a3030',
    opacity: 0.25,
  },
  sheen1: {
    position: 'absolute',
    left: BAG_WIDTH * 0.15,
    top: 28,
    width: 10,
    height: BAG_HEIGHT - 65,
    backgroundColor: '#B03030',
    borderRadius: 5,
    opacity: 0.3,
  },
  sheen2: {
    position: 'absolute',
    left: BAG_WIDTH * 0.35,
    top: 35,
    width: 6,
    height: BAG_HEIGHT * 0.5,
    backgroundColor: '#C04040',
    borderRadius: 3,
    opacity: 0.12,
  },
  bagShadowR: {
    position: 'absolute',
    right: BAG_WIDTH * 0.08,
    top: 28,
    width: 22,
    height: BAG_HEIGHT - 60,
    backgroundColor: '#3a0808',
    borderRadius: 11,
    opacity: 0.3,
  },
  brandPatch: {
    position: 'absolute',
    top: BAG_HEIGHT * 0.4,
    left: BAG_WIDTH * 0.22,
    right: BAG_WIDTH * 0.22,
    height: 32,
    backgroundColor: BAG_DARK,
    borderRadius: 4,
    borderWidth: 0.5,
    borderColor: BAG_DARKER,
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandText: {
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 4,
    color: '#9a3535',
    opacity: 0.7,
  },
  // ── Bottom cap ──
  bagBottom: {
    width: BAG_WIDTH * 0.6,
    height: 16,
    backgroundColor: BAG_DARK,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    alignSelf: 'center',
    marginTop: -2,
    borderWidth: 0.5,
    borderTopWidth: 0,
    borderColor: BAG_DARKER,
  },
  bottomStrap: {
    position: 'absolute',
    top: 5,
    left: 8,
    right: 8,
    height: 2.5,
    backgroundColor: BAG_DARKER,
    borderRadius: 1,
  },
  dRing: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: '#5a5a5a',
    backgroundColor: 'transparent',
    alignSelf: 'center',
    marginTop: 2,
  },
  // ── Bottom row ──
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
