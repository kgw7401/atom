import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';

type Props = {
  navigation: NativeStackNavigationProp<any>;
};

export default function HomeScreen({ navigation }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>ATOM</Text>
      <Text style={styles.subtitle}>AI Boxing Coach</Text>
      <TouchableOpacity
        style={styles.startButton}
        onPress={() => navigation.navigate('SessionSetup')}
      >
        <Text style={styles.startButtonText}>세션 시작</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    color: '#e63946',
    fontSize: 56,
    fontWeight: '900',
    letterSpacing: 8,
  },
  subtitle: {
    color: '#888',
    fontSize: 16,
    marginBottom: 60,
    letterSpacing: 2,
  },
  startButton: {
    backgroundColor: '#e63946',
    borderRadius: 12,
    paddingVertical: 20,
    paddingHorizontal: 60,
  },
  startButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '700',
  },
});
