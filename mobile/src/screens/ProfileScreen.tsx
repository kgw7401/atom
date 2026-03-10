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

type Props = { navigation: NativeStackNavigationProp<any> };

export default function ProfileScreen({ navigation }: Props) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile()
      .then(setProfile)
      .catch(() => Alert.alert('Error', 'Cannot connect to server.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#e63946" />
        <TouchableOpacity style={styles.settingsBtnCenter} onPress={() => navigation.navigate('Settings')}>
          <Text style={styles.settingsBtnText}>⚙ 서버 설정</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!profile) {
    return (
      <View style={styles.center}>
        <Text style={styles.empty}>서버에 연결할 수 없습니다.</Text>
        <TouchableOpacity style={styles.settingsBtnCenter} onPress={() => navigation.navigate('Settings')}>
          <Text style={styles.settingsBtnText}>⚙ 서버 설정</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const topCombos = Object.entries(profile.combo_exposure_json)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <TouchableOpacity style={styles.settingsBtn} onPress={() => navigation.navigate('Settings')}>
        <Text style={styles.settingsBtnText}>⚙ 서버 설정</Text>
      </TouchableOpacity>

      <View style={styles.statRow}>
        <StatBox label="총 세션" value={String(profile.total_sessions)} />
        <StatBox label="훈련 시간" value={`${Math.round(profile.total_training_minutes)}분`} />
        <StatBox label="주당 빈도" value={`${profile.session_frequency.toFixed(1)}회`} />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>레벨</Text>
        <Text style={styles.value}>{profile.experience_level}</Text>
      </View>

      {profile.goal ? (
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>목표</Text>
          <Text style={styles.value}>{profile.goal}</Text>
        </View>
      ) : null}

      {topCombos.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>많이 훈련한 콤보</Text>
          {topCombos.map(([name, count]) => (
            <Text key={name} style={styles.comboRow}>
              {name}  <Text style={styles.comboCount}>{count}회</Text>
            </Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.statBox}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a' },
  content: { padding: 24, paddingBottom: 48 },
  center: { flex: 1, backgroundColor: '#0a0a0a', alignItems: 'center', justifyContent: 'center' },
  empty: { color: '#555', fontSize: 16 },
  settingsBtn: { alignSelf: 'flex-end', padding: 8, marginBottom: 16 },
  settingsBtnCenter: { marginTop: 24, padding: 12, borderWidth: 1, borderColor: '#333', borderRadius: 8 },
  settingsBtnText: { color: '#888', fontSize: 14 },
  statRow: { flexDirection: 'row', gap: 10, marginBottom: 24 },
  statBox: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#222',
  },
  statValue: { color: '#fff', fontSize: 22, fontWeight: '700' },
  statLabel: { color: '#666', fontSize: 11, marginTop: 4 },
  section: { marginBottom: 24 },
  sectionLabel: { color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 },
  value: { color: '#fff', fontSize: 16 },
  comboRow: { color: '#fff', fontSize: 15, marginBottom: 4 },
  comboCount: { color: '#666' },
});
