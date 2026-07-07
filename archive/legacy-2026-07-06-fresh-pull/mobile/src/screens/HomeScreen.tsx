import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View, Text, TouchableOpacity, Pressable, StyleSheet,
  ActivityIndicator, Alert, Dimensions, GestureResponderEvent,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import Animated, {
  useSharedValue, useAnimatedStyle,
  withRepeat, withSequence, withTiming, withSpring, withDelay,
  Easing,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { Audio } from 'expo-av';
import Svg, {
  Defs, LinearGradient, RadialGradient, Stop,
  Path, Rect, Line, Circle, Ellipse,
} from 'react-native-svg';
import { COLORS, TYPOGRAPHY, SPACING } from '../theme';
import { fetchToday, TodayData } from '../api/session';
import { updateProfile } from '../api/profile';

// ── Impact SFX assets ──
const IMPACT_SFX = [
  require('../../assets/sfx/impact/body_1.mp3'),
  require('../../assets/sfx/impact/body_2.mp3'),
  require('../../assets/sfx/impact/body_3.mp3'),
];
const CHAIN_SFX = [
  require('../../assets/sfx/impact/chain_1.mp3'),
  require('../../assets/sfx/impact/chain_2.mp3'),
];

type Props = { navigation: NativeStackNavigationProp<any> };

const { width: SCREEN_W } = Dimensions.get('window');

const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
const LEVEL_LABELS: Record<string, string> = {
  beginner: '입문', novice: '입문', intermediate: '중급', advanced: '고급',
};

// ── Physics reference dimensions ──
const BAG_WIDTH = 110;
const BAG_HEIGHT = 300;

// ── SVG layout constants ──
const SVG_W = 160;
const SVG_H = 440;
const BX = 25;            // body left x
const BW = 110;           // body width
const BT = 88;            // body top y
const BCY = 370;          // where bottom curve begins
const CX = 22;            // collar x
const CW = 116;           // collar width
const CT = 60;            // collar top y
const CH = 30;            // collar height
const CEN = SVG_W / 2;   // center x

// Body + rounded bottom as single closed path
const BAG_BODY = `M${BX},${BT} L${BX + BW},${BT} L${BX + BW},${BCY} A${BW / 2},38,0,0,1,${BX},${BCY} Z`;

// ── Floating particle ──
function FloatingParticle({ delay, x, size }: { delay: number; x: number; size: number }) {
  const translateY = useSharedValue(0);
  const opacity = useSharedValue(0);

  useEffect(() => {
    opacity.value = withDelay(delay,
      withRepeat(withSequence(
        withTiming(0.35, { duration: 2000, easing: Easing.inOut(Easing.sin) }),
        withTiming(0, { duration: 2000, easing: Easing.inOut(Easing.sin) }),
      ), -1, true)
    );
    translateY.value = withDelay(delay,
      withRepeat(withTiming(-120, { duration: 4000, easing: Easing.inOut(Easing.sin) }), -1, true)
    );
  }, []);

  const style = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));

  return (
    <Animated.View
      style={[{
        position: 'absolute', left: x, bottom: '30%',
        width: size, height: size, borderRadius: size / 2,
        backgroundColor: COLORS.RED,
      }, style]}
    />
  );
}

export default function HomeScreen({ navigation }: Props) {
  const [today, setToday] = useState<TodayData | null>(null);
  const [loading, setLoading] = useState(true);

  // ── Impact SFX ──
  const impactSounds = useRef<Audio.Sound[]>([]);
  const chainSounds = useRef<Audio.Sound[]>([]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      await Audio.setAudioModeAsync({ playsInSilentModeIOS: true });
      for (const asset of IMPACT_SFX) {
        try {
          const { sound } = await Audio.Sound.createAsync(asset, { shouldPlay: false });
          if (mounted) impactSounds.current.push(sound);
        } catch (e) { console.warn('Impact SFX load failed:', e); }
      }
      for (const asset of CHAIN_SFX) {
        try {
          const { sound } = await Audio.Sound.createAsync(asset, { shouldPlay: false });
          if (mounted) chainSounds.current.push(sound);
        } catch (e) { console.warn('Chain SFX load failed:', e); }
      }
      console.log(`[HomeScreen] Loaded ${impactSounds.current.length} impact + ${chainSounds.current.length} chain sounds`);
    })();
    return () => {
      mounted = false;
      impactSounds.current.forEach((s) => s.unloadAsync().catch(() => {}));
      chainSounds.current.forEach((s) => s.unloadAsync().catch(() => {}));
    };
  }, []);

  const playImpactSfx = useCallback(async () => {
    const sounds = impactSounds.current;
    if (sounds.length === 0) { console.warn('[HomeScreen] No impact sounds loaded'); return; }
    const hit = sounds[Math.floor(Math.random() * sounds.length)];
    try {
      await hit.setPositionAsync(0);
      await hit.playAsync();
    } catch (e) { console.warn('Impact play failed:', e); }
    // Chain rattle after a short delay
    const chains = chainSounds.current;
    if (chains.length > 0) {
      setTimeout(async () => {
        const chain = chains[Math.floor(Math.random() * chains.length)];
        try {
          await chain.setPositionAsync(0);
          await chain.playAsync();
        } catch {}
      }, 120);
    }
  }, []);

  // ── Shared values ──
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
    // Logo fade-in
    logoOpacity.value = withDelay(200, withTiming(1, { duration: 800 }));
    logoTranslateY.value = withDelay(200, withTiming(0, { duration: 800, easing: Easing.out(Easing.cubic) }));

    // Bag entrance
    bagScale.value = withDelay(400, withSpring(1, { damping: 14, stiffness: 80 }));

    // Idle sway
    bagSwing.value = withDelay(1400,
      withRepeat(withSequence(
        withTiming(1, { duration: 2800, easing: Easing.inOut(Easing.sin) }),
        withTiming(-1, { duration: 2800, easing: Easing.inOut(Easing.sin) }),
      ), -1, true)
    );

    // Glow pulse
    glowOpacity.value = withDelay(600,
      withRepeat(withSequence(
        withTiming(0.5, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
        withTiming(0.15, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
      ), -1, true)
    );
    glowScale.value = withDelay(600,
      withRepeat(withSequence(
        withTiming(1.1, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
        withTiming(0.9, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
      ), -1, true)
    );

    // Bottom badges
    bottomOpacity.value = withDelay(1000, withTiming(1, { duration: 600 }));
    bottomTranslateY.value = withDelay(1000, withTiming(0, { duration: 600, easing: Easing.out(Easing.cubic) }));
  }, []);

  const logoStyle = useAnimatedStyle(() => ({
    opacity: logoOpacity.value,
    transform: [{ translateY: logoTranslateY.value }],
  }));

  const bagContainerStyle = useAnimatedStyle(() => ({
    transform: [
      { scale: bagScale.value },
      { scaleX: bagScaleX.value },
      { scaleY: bagScaleY.value },
      { translateX: bagTranslateX.value },
      { rotate: `${bagSwing.value * 3.5}deg` },
    ],
  }));

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

  // ── Punch handler ──
  const handlePress = (e: GestureResponderEvent) => {
    const tapScreenX = e.nativeEvent.pageX;
    const screenCenterX = SCREEN_W / 2;
    const offsetX = (screenCenterX - tapScreenX) / (SCREEN_W / 4);
    const clampedOffset = Math.max(-1, Math.min(1, offsetX));

    const tapY = e.nativeEvent.locationY;
    const leverArm = Math.max(0.6, 1 - (tapY / (BAG_HEIGHT + 78)) * 0.4);
    const force = Math.max(0.5, Math.abs(clampedOffset) + 0.4) * leverArm;
    const direction = clampedOffset >= 0 ? 1 : -1;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    playImpactSfx();

    // Pendulum: rotation only (pivot at top), no lateral translate
    bagSwing.value = withSpring(0, { velocity: direction * force * 90, damping: 2.5, stiffness: 25, mass: 2 });

    const sq = force * 0.12;
    bagScaleX.value = withSequence(withTiming(1 + sq, { duration: 40 }), withSpring(1, { damping: 6, stiffness: 120 }));
    bagScaleY.value = withSequence(withTiming(1 - sq * 0.8, { duration: 40 }), withSpring(1, { damping: 6, stiffness: 120 }));

    impactFlash.value = withSequence(withTiming(0.6 * force, { duration: 30 }), withTiming(0, { duration: 200 }));
    glowOpacity.value = withSequence(withTiming(0.8, { duration: 50 }), withTiming(0.3, { duration: 600 }));
    glowScale.value = withSequence(withTiming(1.2 + 0.3 * force, { duration: 60 }), withSpring(1, { damping: 8, stiffness: 80 }));

    setTimeout(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium), 60);
    setTimeout(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light), 250);
    setTimeout(() => navigation.navigate('SessionPicker', { today }), 800);
  };

  const handleLevelChange = () => {
    const raw = today?.level ?? 'beginner';
    const current = raw === 'novice' ? 'beginner' : raw;
    const options = LEVELS.filter((l) => l !== current);
    Alert.alert(
      '레벨 변경', `현재: ${LEVEL_LABELS[current]}`,
      [
        ...options.map((lvl) => ({
          text: LEVEL_LABELS[lvl],
          onPress: async () => {
            try {
              await updateProfile({ experience_level: lvl });
              const data = await fetchToday();
              setToday(data);
            } catch (e) { console.error('Level change failed:', e); }
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

      {/* Center: bag assembly */}
      <View style={styles.center}>
        <Animated.View style={[styles.glowOuter, glowStyle]} />
        <Animated.View style={[styles.glowInner, glowStyle]} />

        <Animated.View style={[styles.bagAssembly, bagContainerStyle]}>
          <Pressable onPress={handlePress}>
            <View style={{ width: SVG_W, height: SVG_H }}>
              <Svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`}>
                <Defs>
                  {/* ── Cylindrical body: left-lit 3D gradient ── */}
                  <LinearGradient id="bodyGrad" x1="0" y1="0" x2="1" y2="0">
                    <Stop offset="0" stopColor="#2A0E0C" />
                    <Stop offset="0.06" stopColor="#5A1A18" />
                    <Stop offset="0.18" stopColor="#C84440" />
                    <Stop offset="0.28" stopColor="#E85550" />
                    <Stop offset="0.42" stopColor="#C83733" />
                    <Stop offset="0.58" stopColor="#B02E2B" />
                    <Stop offset="0.75" stopColor="#7A2220" />
                    <Stop offset="0.90" stopColor="#4A1515" />
                    <Stop offset="1" stopColor="#1E0A08" />
                  </LinearGradient>

                  {/* ── Specular highlight (glossy leather) ── */}
                  <RadialGradient id="specular" cx="0.30" cy="0.25" rx="0.30" ry="0.22">
                    <Stop offset="0" stopColor="#ffffff" stopOpacity="0.30" />
                    <Stop offset="0.5" stopColor="#ffffff" stopOpacity="0.08" />
                    <Stop offset="1" stopColor="#ffffff" stopOpacity="0" />
                  </RadialGradient>

                  {/* ── Secondary rim light (right edge) ── */}
                  <LinearGradient id="rimLight" x1="1" y1="0" x2="0" y2="0">
                    <Stop offset="0" stopColor="#FF8A80" stopOpacity="0.15" />
                    <Stop offset="0.08" stopColor="#FF8A80" stopOpacity="0.06" />
                    <Stop offset="0.15" stopColor="#FF8A80" stopOpacity="0" />
                  </LinearGradient>

                  {/* ── Collar depth gradient ── */}
                  <LinearGradient id="collarGrad" x1="0" y1="0" x2="0" y2="1">
                    <Stop offset="0" stopColor="#3a3a3a" />
                    <Stop offset="0.35" stopColor="#1e1e1e" />
                    <Stop offset="1" stopColor="#0a0a0a" />
                  </LinearGradient>

                  {/* ── Collar cylindrical shading ── */}
                  <LinearGradient id="collarSide" x1="0" y1="0" x2="1" y2="0">
                    <Stop offset="0" stopColor="#0a0a0a" />
                    <Stop offset="0.15" stopColor="#333" />
                    <Stop offset="0.45" stopColor="#444" />
                    <Stop offset="0.85" stopColor="#222" />
                    <Stop offset="1" stopColor="#080808" />
                  </LinearGradient>

                  {/* ── Metallic chain gradient ── */}
                  <LinearGradient id="chainGrad" x1="0" y1="0" x2="0" y2="1">
                    <Stop offset="0" stopColor="#999" />
                    <Stop offset="0.3" stopColor="#666" />
                    <Stop offset="0.6" stopColor="#888" />
                    <Stop offset="1" stopColor="#555" />
                  </LinearGradient>

                  {/* ── Ground shadow ── */}
                  <RadialGradient id="shadowGrad" cx="0.5" cy="0.5" rx="0.5" ry="0.5">
                    <Stop offset="0" stopColor="#000" stopOpacity="0.30" />
                    <Stop offset="0.6" stopColor="#000" stopOpacity="0.08" />
                    <Stop offset="1" stopColor="#000" stopOpacity="0" />
                  </RadialGradient>
                </Defs>

                {/* ── Ground shadow ── */}
                <Ellipse cx={CEN} cy={SVG_H - 8} rx={55} ry={8} fill="url(#shadowGrad)" />

                {/* ── Mount bracket ── */}
                <Rect x={CEN - 14} y={0} width={28} height={12} rx={4} fill="#3a3a3a" />
                <Rect x={CEN - 10} y={3} width={20} height={6} rx={2} fill="#505050" />
                {/* Mount bolt */}
                <Circle cx={CEN} cy={6} r={2.5} fill="#666" />

                {/* ── Chains — V-pattern with center ── */}
                <Line x1={CEN - 8} y1={12} x2={CX + 16} y2={CT + 5}
                  stroke="url(#chainGrad)" strokeWidth={2.5} strokeLinecap="round" />
                <Line x1={CEN + 8} y1={12} x2={CX + CW - 16} y2={CT + 5}
                  stroke="url(#chainGrad)" strokeWidth={2.5} strokeLinecap="round" />
                <Line x1={CEN} y1={12} x2={CEN} y2={CT + 3}
                  stroke="url(#chainGrad)" strokeWidth={2} strokeLinecap="round" />

                {/* ── Collar ── */}
                <Rect x={CX} y={CT} width={CW} height={CH} rx={5} fill="url(#collarSide)" />
                {/* Collar band detail */}
                <Rect x={CX + 3} y={CT + CH - 7} width={CW - 6} height={3} rx={1.5} fill="#333" />
                {/* Collar rivets */}
                <Circle cx={CX + 12} cy={CT + 13} r={2.5} fill="#3a3a3a" />
                <Circle cx={CX + 12} cy={CT + 13} r={1.2} fill="#555" />
                <Circle cx={CX + CW - 12} cy={CT + 13} r={2.5} fill="#3a3a3a" />
                <Circle cx={CX + CW - 12} cy={CT + 13} r={1.2} fill="#555" />

                {/* ── Ambient occlusion: collar → body junction ── */}
                <Rect x={BX} y={BT} width={BW} height={8} fill="#1a0808" opacity={0.5} />

                {/* ── Bag body (3D cylinder) ── */}
                <Path d={BAG_BODY} fill="url(#bodyGrad)" />

                {/* ── Specular highlight overlay ── */}
                <Path d={BAG_BODY} fill="url(#specular)" />

                {/* ── Rim light (right edge catch light) ── */}
                <Path d={BAG_BODY} fill="url(#rimLight)" />

                {/* ── Vertical seam ── */}
                <Line x1={CEN} y1={BT + 12} x2={CEN} y2={BCY - 10}
                  stroke="#8B2724" strokeWidth={0.7} opacity={0.25} />

                {/* ── Bottom highlight (curved surface catching light) ── */}
                <Ellipse cx={CEN - 10} cy={BCY + 10} rx={22} ry={8}
                  fill="#E85550" opacity={0.12} />
              </Svg>

              {/* Impact flash overlay (positioned over bag body) */}
              <Animated.View style={[styles.flashOverlay, impactFlashStyle]} pointerEvents="none" />
            </View>
          </Pressable>
        </Animated.View>
      </View>

      {/* Bottom badges */}
      <Animated.View style={[styles.bottomRow, bottomStyle]}>
        <TouchableOpacity style={styles.levelBadge} onPress={handleLevelChange}>
          <Text style={styles.levelText}>{LEVEL_LABELS[today?.level ?? 'beginner']}</Text>
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

const styles = StyleSheet.create({
  container: {
    flex: 1, backgroundColor: COLORS.BG,
    paddingHorizontal: SPACING.PADDING_SCREEN,
    paddingTop: 70, paddingBottom: 44, alignItems: 'center',
  },
  logo: {
    fontSize: 36, fontWeight: '900', letterSpacing: 10,
    color: COLORS.RED, textAlign: 'center',
  },
  center: {
    flex: 1, justifyContent: 'center', alignItems: 'center',
  },
  glowOuter: {
    position: 'absolute', width: 260, height: 260, borderRadius: 130,
    backgroundColor: 'transparent', borderWidth: 1, borderColor: COLORS.RED,
  },
  glowInner: {
    position: 'absolute', width: 200, height: 200, borderRadius: 100,
    backgroundColor: COLORS.RED, opacity: 0.15,
  },
  bagAssembly: {
    alignItems: 'center', transformOrigin: 'top center',
  },
  flashOverlay: {
    position: 'absolute',
    top: BT,
    left: BX,
    width: BW,
    height: BCY - BT + 38,
    backgroundColor: '#fff',
    borderBottomLeftRadius: BW / 2,
    borderBottomRightRadius: BW / 2,
  },
  bottomRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
  },
  levelBadge: {
    backgroundColor: COLORS.SURFACE,
    paddingHorizontal: 14, paddingVertical: 6,
    borderRadius: SPACING.RADIUS_BADGE,
    borderWidth: 1, borderColor: COLORS.TEXT_3,
  },
  levelText: {
    color: COLORS.TEXT_2, ...TYPOGRAPHY.META, fontWeight: '700',
  },
  streakBadge: {
    backgroundColor: COLORS.SURFACE,
    paddingHorizontal: 14, paddingVertical: 6,
    borderRadius: SPACING.RADIUS_BADGE,
    borderWidth: 1, borderColor: COLORS.ORANGE,
  },
  streakText: {
    color: COLORS.ORANGE, ...TYPOGRAPHY.META, fontWeight: '700',
  },
});
