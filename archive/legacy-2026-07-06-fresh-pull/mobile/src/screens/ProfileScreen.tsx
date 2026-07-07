import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Alert,
  TouchableOpacity,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { fetchProfile, Profile } from '../api/profile';
import { COLORS, SPACING } from '../theme';

type Props = { navigation: NativeStackNavigationProp<any> };

const LEVEL_LABELS: Record<string, string> = {
  beginner: '입문',
  novice: '초급',
  intermediate: '중급',
  advanced: '고급',
};

export default function ProfileScreen({ navigation }: Props) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile()
      .then(setProfile)
      .catch(() => Alert.alert('오류', '서버에 연결할 수 없습니다.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={COLORS.RED} />
      </View>
    );
  }

  if (!profile) {
    return (
      <View style={styles.center}>
        <Text style={styles.empty}>서버에 연결할 수 없습니다.</Text>
        <TouchableOpacity
          style={styles.settingsBtn}
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.settingsBtnText}>서버 설정</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Settings */}
      <TouchableOpacity style={styles.gearBtn} onPress={() => navigation.navigate('Settings')}>
        <Text style={styles.gearText}>⚙</Text>
      </TouchableOpacity>

      {/* Level */}
      <View style={styles.levelRow}>
        <Text style={styles.levelLabel}>레벨</Text>
        <Text style={styles.levelValue}>
          {LEVEL_LABELS[profile.experience_level] ?? profile.experience_level}
        </Text>
      </View>

      {/* Stats */}
      <View style={styles.statsRow}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{profile.total_sessions}</Text>
          <Text style={styles.statLabel}>총 세션</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{Math.round(profile.total_training_minutes)}분</Text>
          <Text style={styles.statLabel}>훈련 시간</Text>
        </View>
      </View>

      {/* Goal */}
      {!!profile.goal && (
        <View style={styles.goalCard}>
          <Text style={styles.goalLabel}>목표</Text>
          <Text style={styles.goalText}>{profile.goal}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.BG },
  content: { padding: SPACING.PADDING_SCREEN, paddingBottom: 48 },
  center: { flex: 1, backgroundColor: COLORS.BG, alignItems: 'center', justifyContent: 'center' },
  empty: { color: COLORS.TEXT_3, fontSize: 16 },
  settingsBtn: {
    marginTop: 24,
    padding: 12,
    borderWidth: 1,
    borderColor: COLORS.TEXT_GHOST,
    borderRadius: 8,
  },
  settingsBtnText: { color: COLORS.TEXT_2, fontSize: 14 },

  gearBtn: { alignSelf: 'flex-end', padding: 4, marginBottom: 16 },
  gearText: { fontSize: 20, color: COLORS.TEXT_2 },

  levelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  levelLabel: { color: COLORS.TEXT_3, fontSize: 14 },
  levelValue: { color: COLORS.TEXT_1, fontSize: 15, fontWeight: '700', textTransform: 'capitalize' },

  statsRow: {
    flexDirection: 'row',
    backgroundColor: COLORS.SURFACE,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    borderRadius: 14,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
  },
  statItem: { flex: 1, alignItems: 'center' },
  statValue: { color: COLORS.TEXT_1, fontSize: 22, fontWeight: '700' },
  statLabel: { color: COLORS.TEXT_3, fontSize: 12, marginTop: 4 },
  statDivider: { width: 1, height: 36, backgroundColor: COLORS.BORDER },

  goalCard: {
    backgroundColor: COLORS.SURFACE,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    borderRadius: 14,
    padding: 20,
  },
  goalLabel: { color: COLORS.TEXT_3, fontSize: 12, marginBottom: 6 },
  goalText: { color: COLORS.TEXT_1, fontSize: 15 },
});
