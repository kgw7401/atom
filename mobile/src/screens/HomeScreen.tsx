import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, TouchableOpacity, Pressable, StyleSheet, ActivityIndicator, Alert, Dimensions, GestureResponderEvent } from 'react-native';
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
import * as Haptics from 'expo-haptics';
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
  const bagTranslateX = useSharedValue(0);
  const bagScale = useSharedValue(0);
  const bagScaleX = useSharedValue(1);
  const bagScaleY = useSharedValue(1);
  const impactFlash = useSharedValue(0);
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
    const rotate = bagSwing.value * 1.8;
    return {
      transform: [
        { scale: bagScale.value },
        { scaleX: bagScaleX.value },
        { scaleY: bagScaleY.value },
        { translateX: bagTranslateX.value },
        { rotate: `${rotate}deg` },
      ],
    };
  });

  const impactFlashStyle = useAnimatedStyle(() => ({
    opacity: impactFlash.value,
  }));

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
    transform: [{ scale: glowScale.value }],
  }));

  const bottomStyle = useAnimatedStyle(() => ({
    opacity: bottomOpacity.value,
    transform: [{ translateY: bottomTranslateY.value }],
  }));

  const handlePress = (e: GestureResponderEvent) => {
    // Use absolute screen position for reliable direction detection
    const tapScreenX = e.nativeEvent.pageX;
    const screenCenterX = SCREEN_W / 2;

    // Punch from left → bag goes right (positive), punch from right → left (negative)
    const offsetX = (screenCenterX - tapScreenX) / (SCREEN_W / 4); // normalized, clamped
    const clampedOffset = Math.max(-1, Math.min(1, offsetX));

    // Tap Y for lever arm (higher = more torque)
    const tapY = e.nativeEvent.locationY;
    const leverArm = Math.max(0.6, 1 - (tapY / (BAG_HEIGHT + 78)) * 0.4);

    // Force from distance to center
    const force = Math.max(0.5, Math.abs(clampedOffset) + 0.4) * leverArm;
    const direction = clampedOffset >= 0 ? 1 : -1;

    // ── Impulse velocity (no position tween — pure physics) ──
    // Apply instant velocity, let spring physics handle oscillation
    const swingVelocity = direction * force * 45;
    const translateVelocity = direction * force * 60;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

    // Rotation: impulse → pendulum oscillation with heavy mass
    bagSwing.value = withSpring(0, {
      velocity: swingVelocity,
      damping: 2.5,
      stiffness: 25,
      mass: 2,
    });

    // Lateral: impulse → arc movement synced with rotation
    bagTranslateX.value = withSpring(0, {
      velocity: translateVelocity,
      damping: 2.5,
      stiffness: 25,
      mass: 2,
    });

    // Squash on impact — quick deformation then spring settle
    const squashForce = force * 0.12;
    bagScaleX.value = withSequence(
      withTiming(1 + squashForce, { duration: 40 }),
      withSpring(1, { damping: 6, stiffness: 120 }),
    );
    bagScaleY.value = withSequence(
      withTiming(1 - squashForce * 0.8, { duration: 40 }),
      withSpring(1, { damping: 6, stiffness: 120 }),
    );

    // Impact flash
    impactFlash.value = withSequence(
      withTiming(0.6 * force, { duration: 30 }),
      withTiming(0, { duration: 200 }),
    );

    // Glow burst
    glowOpacity.value = withSequence(
      withTiming(0.8, { duration: 50 }),
      withTiming(0.3, { duration: 600 }),
    );
    glowScale.value = withSequence(
      withTiming(1.2 + 0.3 * force, { duration: 60 }),
      withSpring(1, { damping: 8, stiffness: 80 }),
    );

    // Follow-through haptics
    setTimeout(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium), 60);
    setTimeout(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light), 250);

    // Navigate after physics play out
    setTimeout(() => navigation.navigate('SessionPicker', { today }), 800);
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
          <Pressable style={styles.bagButton} onPress={handlePress}>
            {/* Chains converging to top */}
            <View style={styles.chainArea}>
              <View style={[styles.chain, styles.chainL]} />
              <View style={[styles.chain, styles.chainR]} />
            </View>

            {/* Black top collar with straps */}
            <View style={styles.collar}>
              <View style={styles.collarStrapL} />
              <View style={styles.collarStrapR} />
              <View style={styles.collarBand} />
            </View>

            {/* Bag body — cylinder */}
            <View style={styles.bagBody}>
              {/* Impact flash */}
              <Animated.View style={[styles.impactFlash, impactFlashStyle]} />
              {/* Shading: left highlight → center → right shadow */}
              <View style={styles.highlightEdge} />
              <View style={styles.highlightMain} />
              <View style={styles.shadowMid} />
              <View style={styles.shadowEdge} />
              {/* Subtle vertical seam */}
              <View style={styles.seam} />
            </View>

            {/* Bottom rounded end */}
            <View style={styles.bagBottom} />
          </Pressable>
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

const BAG_WIDTH = 110;
const BAG_HEIGHT = 300;
const BAG_RED = '#B5332E';      // main leather red (like reference)
const BAG_RED_LIGHT = '#D04540'; // highlight
const BAG_RED_DARK = '#7A2220';  // shadow
const COLLAR_COLOR = '#1a1a1a'; // black leather collar

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
  // ── Chains ──
  chainArea: {
    width: BAG_WIDTH,
    height: 50,
    position: 'relative',
  },
  chain: {
    position: 'absolute',
    width: 3,
    backgroundColor: '#3d3d3d',
    borderRadius: 1.5,
  },
  chainL: {
    top: 0,
    left: BAG_WIDTH / 2 - 1.5,
    height: 50,
    transform: [{ rotate: '14deg' }],
    transformOrigin: 'top center',
  },
  chainR: {
    top: 0,
    right: BAG_WIDTH / 2 - 1.5,
    height: 50,
    transform: [{ rotate: '-14deg' }],
    transformOrigin: 'top center',
  },
  // ── Black collar ──
  collar: {
    width: BAG_WIDTH + 4,
    height: 28,
    backgroundColor: COLLAR_COLOR,
    borderRadius: 6,
    marginTop: -4,
    zIndex: 2,
    overflow: 'hidden',
  },
  collarStrapL: {
    position: 'absolute',
    top: -10,
    left: BAG_WIDTH * 0.3,
    width: 12,
    height: 24,
    backgroundColor: '#2a2a2a',
    transform: [{ rotate: '5deg' }],
  },
  collarStrapR: {
    position: 'absolute',
    top: -10,
    right: BAG_WIDTH * 0.3,
    width: 12,
    height: 24,
    backgroundColor: '#2a2a2a',
    transform: [{ rotate: '-5deg' }],
  },
  collarBand: {
    position: 'absolute',
    bottom: 4,
    left: 6,
    right: 6,
    height: 3,
    backgroundColor: '#333',
    borderRadius: 1.5,
  },
  // ── Impact flash ──
  impactFlash: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#fff',
    zIndex: 10,
  },
  // ── Bag body (cylinder) ──
  bagBody: {
    width: BAG_WIDTH,
    height: BAG_HEIGHT,
    backgroundColor: BAG_RED,
    overflow: 'hidden',
    marginTop: -2,
  },
  highlightEdge: {
    position: 'absolute',
    left: 0,
    top: 0,
    width: BAG_WIDTH * 0.12,
    height: BAG_HEIGHT,
    backgroundColor: BAG_RED_DARK,
    opacity: 0.4,
  },
  highlightMain: {
    position: 'absolute',
    left: BAG_WIDTH * 0.18,
    top: 0,
    width: BAG_WIDTH * 0.22,
    height: BAG_HEIGHT,
    backgroundColor: BAG_RED_LIGHT,
    opacity: 0.25,
  },
  shadowMid: {
    position: 'absolute',
    right: BAG_WIDTH * 0.12,
    top: 0,
    width: BAG_WIDTH * 0.2,
    height: BAG_HEIGHT,
    backgroundColor: BAG_RED_DARK,
    opacity: 0.2,
  },
  shadowEdge: {
    position: 'absolute',
    right: 0,
    top: 0,
    width: BAG_WIDTH * 0.14,
    height: BAG_HEIGHT,
    backgroundColor: '#4a1515',
    opacity: 0.5,
  },
  seam: {
    position: 'absolute',
    left: BAG_WIDTH / 2 - 0.5,
    top: 0,
    width: 1,
    height: BAG_HEIGHT,
    backgroundColor: BAG_RED_DARK,
    opacity: 0.3,
  },
  // ── Bottom ──
  bagBottom: {
    width: BAG_WIDTH,
    height: BAG_WIDTH / 2,
    backgroundColor: BAG_RED,
    borderBottomLeftRadius: BAG_WIDTH / 2,
    borderBottomRightRadius: BAG_WIDTH / 2,
    marginTop: -1,
    overflow: 'hidden',
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
