import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { generatePlan } from '../api/session';
import { COLORS, SPACING } from '../theme';

type Props = { navigation: NativeStackNavigationProp<any> };

const LEVEL_OPTIONS = [
  { label: '초급', value: 'beginner' as const },
  { label: '중급', value: 'intermediate' as const },
  { label: '고급', value: 'advanced' as const },
];
const ROUND_OPTIONS = [1, 2, 3, 4, 5, 6];
const DURATION_OPTIONS = [
  { label: '1분', value: 60 },
  { label: '2분', value: 120 },
  { label: '3분', value: 180 },
];
const REST_OPTIONS = [
  { label: '15초', value: 15 },
  { label: '30초', value: 30 },
  { label: '45초', value: 45 },
  { label: '60초', value: 60 },
];

export default function SessionSetupScreen({ navigation }: Props) {
  const [level, setLevel] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner');
  const [rounds, setRounds] = useState(3);
  const [roundDuration, setRoundDuration] = useState(180);
  const [restDuration, setRestDuration] = useState(30);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const result = await generatePlan({
        level,
        rounds,
        round_duration_sec: roundDuration,
        rest_sec: restDuration,
      });
      navigation.navigate('PlanPreview', { result });
    } catch (e: any) {
      Alert.alert('오류', e.message ?? '플랜 생성에 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      {/* Rounds */}
      <View style={styles.section}>
        <Text style={styles.label}>라운드 수</Text>
        <View style={styles.chips}>
          {ROUND_OPTIONS.map((r) => (
            <TouchableOpacity
              key={r}
              style={[styles.chip, rounds === r && styles.chipActive]}
              onPress={() => setRounds(r)}
              activeOpacity={0.7}
            >
              <Text style={[styles.chipText, rounds === r && styles.chipTextActive]}>
                {r}R
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Round duration */}
      <View style={styles.section}>
        <Text style={styles.label}>라운드 시간</Text>
        <View style={styles.chips}>
          {DURATION_OPTIONS.map((d) => (
            <TouchableOpacity
              key={d.value}
              style={[styles.chip, roundDuration === d.value && styles.chipActive]}
              onPress={() => setRoundDuration(d.value)}
              activeOpacity={0.7}
            >
              <Text style={[styles.chipText, roundDuration === d.value && styles.chipTextActive]}>
                {d.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Rest duration */}
      <View style={styles.section}>
        <Text style={styles.label}>휴식 시간</Text>
        <View style={styles.chips}>
          {REST_OPTIONS.map((r) => (
            <TouchableOpacity
              key={r.value}
              style={[styles.chip, restDuration === r.value && styles.chipActive]}
              onPress={() => setRestDuration(r.value)}
              activeOpacity={0.7}
            >
              <Text style={[styles.chipText, restDuration === r.value && styles.chipTextActive]}>
                {r.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Level */}
      <View style={styles.section}>
        <Text style={styles.label}>난이도</Text>
        <View style={styles.chips}>
          {LEVEL_OPTIONS.map((l) => (
            <TouchableOpacity
              key={l.value}
              style={[styles.chip, level === l.value && styles.chipActive]}
              onPress={() => setLevel(l.value)}
              activeOpacity={0.7}
            >
              <Text style={[styles.chipText, level === l.value && styles.chipTextActive]}>
                {l.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Generate button */}
      <TouchableOpacity
        style={[styles.generateBtn, loading && styles.generateBtnDisabled]}
        onPress={handleGenerate}
        disabled={loading}
        activeOpacity={0.85}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.generateBtnText}>세션 생성</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.BG },
  content: { padding: SPACING.PADDING_SCREEN, paddingBottom: 48 },

  section: { marginBottom: 28 },
  label: {
    color: COLORS.TEXT_2,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    marginBottom: 12,
  },

  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 20,
  },
  chipActive: {
    backgroundColor: COLORS.RED,
    borderColor: COLORS.RED,
  },
  chipText: {
    color: COLORS.TEXT_2,
    fontSize: 15,
    fontWeight: '600',
  },
  chipTextActive: {
    color: '#fff',
  },

  generateBtn: {
    backgroundColor: COLORS.RED,
    borderRadius: 14,
    paddingVertical: 18,
    alignItems: 'center',
    marginTop: 8,
  },
  generateBtnDisabled: {
    opacity: 0.5,
  },
  generateBtnText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
  },
});
