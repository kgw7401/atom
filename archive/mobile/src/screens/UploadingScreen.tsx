import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { ApiClient } from '../services/api';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Uploading'>;

export default function UploadingScreen({ route, navigation }: Props) {
  const { sessionId, videoUri } = route.params;
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [status, setStatus] = useState<'uploading' | 'analyzing' | 'completed' | 'failed'>(
    'uploading'
  );

  useEffect(() => {
    uploadVideo();
  }, []);

  const uploadVideo = async () => {
    try {
      // Upload video
      await ApiClient.uploadVideo(sessionId, videoUri, (progress) => {
        setUploadProgress(progress);
      });

      setStatus('analyzing');
      pollAnalysisStatus();
    } catch (error) {
      console.error('Upload error:', error);
      setStatus('failed');
      Alert.alert('Error', 'Failed to upload video. Please try again.');
    }
  };

  const pollAnalysisStatus = async () => {
    try {
      const checkStatus = async () => {
        const result = await ApiClient.getSessionStatus(sessionId);
        setAnalysisProgress(result.progress * 100);

        if (result.status === 'completed') {
          setStatus('completed');
          // Navigate to report after short delay
          setTimeout(() => {
            navigation.replace('Report', { sessionId });
          }, 1000);
        } else if (result.status === 'failed') {
          setStatus('failed');
          Alert.alert('Error', 'Analysis failed. Please try again.');
        } else {
          // Keep polling
          setTimeout(checkStatus, 2000);
        }
      };

      checkStatus();
    } catch (error) {
      console.error('Polling error:', error);
      setStatus('failed');
      Alert.alert('Error', 'Failed to check analysis status');
    }
  };

  const getStatusMessage = () => {
    switch (status) {
      case 'uploading':
        return 'Uploading video...';
      case 'analyzing':
        return 'Analyzing your performance...';
      case 'completed':
        return 'Analysis complete!';
      case 'failed':
        return 'Analysis failed';
    }
  };

  const getProgressValue = () => {
    if (status === 'uploading') return uploadProgress;
    if (status === 'analyzing') return analysisProgress;
    if (status === 'completed') return 100;
    return 0;
  };

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#e74c3c" />
      <Text style={styles.title}>{getStatusMessage()}</Text>

      <View style={styles.progressContainer}>
        <View style={styles.progressBar}>
          <View
            style={[styles.progressFill, { width: `${getProgressValue()}%` }]}
          />
        </View>
        <Text style={styles.progressText}>{Math.round(getProgressValue())}%</Text>
      </View>

      {status === 'analyzing' && (
        <View style={styles.infoBox}>
          <Text style={styles.infoText}>
            Our AI is analyzing your boxing technique using computer vision and deep learning.
          </Text>
          <Text style={styles.infoText}>This may take 30-60 seconds.</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 30,
    textAlign: 'center',
  },
  progressContainer: {
    width: '100%',
    marginTop: 40,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#333',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#e74c3c',
  },
  progressText: {
    color: '#888',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 10,
  },
  infoBox: {
    marginTop: 40,
    padding: 20,
    backgroundColor: '#111',
    borderRadius: 10,
  },
  infoText: {
    color: '#888',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 8,
  },
});
