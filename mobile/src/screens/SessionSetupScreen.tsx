import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  TextInput,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { fetchTemplates, Template } from '../api/templates';
import { generatePlan } from '../api/session';

type Props = {
  navigation: NativeStackNavigationProp<any>;
};

const TEMPLATE_LABELS: Record<string, string> = {
  fundamentals: '기본기',
  combos: '콤비네이션',
  mixed: '종합',
};

export default function SessionSetupScreen({ navigation }: Props) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selected, setSelected] = useState<string>('fundamentals');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchTemplates()
      .then(setTemplates)
      .catch(() => Alert.alert('Error', 'Cannot connect to server. Check Settings.'));
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const result = await generatePlan({ template: selected, prompt: prompt || undefined });
      navigation.navigate('PlanPreview', { result });
    } catch (e: any) {
      Alert.alert('Error', e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionLabel}>템플릿 선택</Text>
      {(['fundamentals', 'combos', 'mixed'] as const).map((name) => {
        const t = templates.find((x) => x.name === name);
        return (
          <TouchableOpacity
            key={name}
            style={[styles.templateCard, selected === name && styles.templateSelected]}
            onPress={() => setSelected(name)}
          >
            <Text style={styles.templateName}>{TEMPLATE_LABELS[name]}</Text>
            {t && <Text style={styles.templateDesc}>{t.description}</Text>}
          </TouchableOpacity>
        );
      })}

      <Text style={[styles.sectionLabel, { marginTop: 32 }]}>요청사항 (선택)</Text>
      <TextInput
        style={styles.promptInput}
        value={prompt}
        onChangeText={setPrompt}
        placeholder="예: 잽 크로스 위주로, 바디샷 많이"
        placeholderTextColor="#555"
        multiline
      />

      <TouchableOpacity
        style={[styles.button, loading && styles.buttonDisabled]}
        onPress={handleGenerate}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>플랜 생성</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a' },
  content: { padding: 24, paddingBottom: 48 },
  sectionLabel: { color: '#888', fontSize: 13, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 },
  templateCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#222',
  },
  templateSelected: { borderColor: '#e63946', backgroundColor: '#1f0a0c' },
  templateName: { color: '#fff', fontSize: 17, fontWeight: '600', marginBottom: 4 },
  templateDesc: { color: '#888', fontSize: 13 },
  promptInput: {
    backgroundColor: '#1a1a1a',
    color: '#fff',
    borderRadius: 8,
    padding: 14,
    fontSize: 15,
    borderWidth: 1,
    borderColor: '#333',
    minHeight: 80,
    textAlignVertical: 'top',
    marginBottom: 24,
  },
  button: {
    backgroundColor: '#e63946',
    borderRadius: 10,
    padding: 18,
    alignItems: 'center',
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '700' },
});
