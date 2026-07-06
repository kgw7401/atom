import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ApiClient } from '../services/api';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  const [userId, setUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    initializeUser();
  }, []);

  const initializeUser = async () => {
    try {
      // Check if user already exists locally
      let storedUserId = await AsyncStorage.getItem('userId');

      if (!storedUserId) {
        // Create new user with device ID
        const deviceId = await getDeviceId();
        const user = await ApiClient.createUser(deviceId);
        storedUserId = user.user_id;
        await AsyncStorage.setItem('userId', storedUserId);
      }

      setUserId(storedUserId);
    } catch (error) {
      console.error('Failed to initialize user:', error);
      Alert.alert('Error', 'Failed to initialize user');
    } finally {
      setLoading(false);
    }
  };

  const getDeviceId = async (): Promise<string> => {
    // Try to get existing device ID or generate new one
    let deviceId = await AsyncStorage.getItem('deviceId');
    if (!deviceId) {
      // Generate simple UUID-like device ID
      deviceId = `mobile-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      await AsyncStorage.setItem('deviceId', deviceId);
    }
    return deviceId;
  };

  const handleStartSession = () => {
    if (!userId) {
      Alert.alert('Error', 'User not initialized');
      return;
    }
    navigation.navigate('SessionSetup', { userId });
  };

  const handleViewTwin = () => {
    if (!userId) {
      Alert.alert('Error', 'User not initialized');
      return;
    }
    navigation.navigate('Twin', { userId });
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ATOM Boxing Coach</Text>
      <Text style={styles.subtitle}>Your Digital Boxing Twin</Text>

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.primaryButton} onPress={handleStartSession}>
          <Text style={styles.primaryButtonText}>Start New Session</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryButton} onPress={handleViewTwin}>
          <Text style={styles.secondaryButtonText}>View Digital Twin</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.infoSection}>
        <Text style={styles.infoTitle}>How it works:</Text>
        <Text style={styles.infoText}>1. Choose your level and duration</Text>
        <Text style={styles.infoText}>2. Follow audio instructions while recording</Text>
        <Text style={styles.infoText}>3. Get instant AI-powered feedback</Text>
        <Text style={styles.infoText}>4. Track your progress over time</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    padding: 20,
    justifyContent: 'center',
  },
  loadingText: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    color: '#888',
    textAlign: 'center',
    marginBottom: 60,
  },
  buttonContainer: {
    marginBottom: 40,
  },
  primaryButton: {
    backgroundColor: '#e74c3c',
    padding: 20,
    borderRadius: 10,
    marginBottom: 15,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  secondaryButton: {
    backgroundColor: '#333',
    padding: 20,
    borderRadius: 10,
  },
  secondaryButtonText: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
  },
  infoSection: {
    marginTop: 20,
    padding: 20,
    backgroundColor: '#111',
    borderRadius: 10,
  },
  infoTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  infoText: {
    color: '#888',
    fontSize: 14,
    marginBottom: 8,
  },
});
