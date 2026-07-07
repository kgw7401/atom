import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useSettingsStore } from '../store/settingsStore';
import { COLORS, SPACING, TYPOGRAPHY } from '../theme';
import SectionLabel from '../components/SectionLabel';
import PrimaryButton from '../components/PrimaryButton';
import SecondaryButton from '../components/SecondaryButton';

export default function SettingsScreen() {
  const { serverUrl, setServerUrl } = useSettingsStore();
  const [input, setInput] = useState(serverUrl);
  const [testResult, setTestResult] = useState<'success' | 'fail' | null>(null);
  const [testing, setTesting] = useState(false);

  const save = async () => {
    if (!input.startsWith('http')) {
      Alert.alert('오류', 'http:// 또는 https://로 시작해야 합니다.');
      return;
    }
    await setServerUrl(input);
    Alert.alert('저장됨', `서버 주소: ${input}`);
  };

  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const url = input.replace(/\/+$/, '');
      const res = await fetch(`${url}/api/profile`, { method: 'GET' });
      setTestResult(res.ok ? 'success' : 'fail');
    } catch {
      setTestResult('fail');
    } finally {
      setTesting(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <SectionLabel text="백엔드 서버 주소" />
      <TextInput
        style={styles.input}
        value={input}
        onChangeText={(t) => { setInput(t); setTestResult(null); }}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="http://192.168.x.x:8000"
        placeholderTextColor={COLORS.TEXT_GHOST}
      />
      <Text style={styles.hint}>
        컴퓨터에서 atom serve 실행 후{'\n'}로컬 IP를 입력하세요.{'\n'}
        예: http://192.168.x.x:8000
      </Text>

      <View style={styles.buttons}>
        <PrimaryButton label="저장" onPress={save} />
        <View style={{ height: 12 }} />
        <SecondaryButton label={testing ? '테스트 중...' : '연결 테스트'} onPress={testConnection} />
      </View>

      {testResult && (
        <Text style={[styles.result, { color: testResult === 'success' ? COLORS.GREEN : COLORS.RED }]}>
          {testResult === 'success' ? '✓ 연결됨' : '✗ 연결 실패'}
        </Text>
      )}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.BG,
    padding: SPACING.PADDING_SCREEN,
    paddingTop: 16,
  },
  input: {
    backgroundColor: COLORS.SURFACE,
    color: COLORS.TEXT_1,
    borderRadius: SPACING.RADIUS_CARD,
    padding: SPACING.PADDING_CARD,
    fontSize: 16,
    borderWidth: 1,
    borderColor: COLORS.BORDER,
    marginBottom: 12,
  },
  hint: {
    color: COLORS.TEXT_3,
    fontSize: 12,
    lineHeight: 18,
    marginBottom: SPACING.GAP_SECTION,
  },
  buttons: { marginBottom: 16 },
  result: {
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
});
