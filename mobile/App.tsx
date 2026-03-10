import React, { useEffect } from 'react';
import { setAudioModeAsync } from 'expo-audio';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { Text } from 'react-native';

import { useSettingsStore } from './src/store/settingsStore';
import HomeScreen from './src/screens/HomeScreen';
import SessionSetupScreen from './src/screens/SessionSetupScreen';
import PlanPreviewScreen from './src/screens/PlanPreviewScreen';
import ActiveSessionScreen from './src/screens/ActiveSessionScreen';
import SessionEndScreen from './src/screens/SessionEndScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const DARK = {
  headerStyle: { backgroundColor: '#0a0a0a' },
  headerTintColor: '#fff',
  headerTitleStyle: { fontWeight: '700' as const },
  contentStyle: { backgroundColor: '#0a0a0a' },
};

function HomeStack() {
  return (
    <Stack.Navigator screenOptions={DARK}>
      <Stack.Screen name="Home" component={HomeScreen} options={{ title: 'ATOM' }} />
      <Stack.Screen name="SessionSetup" component={SessionSetupScreen} options={{ title: '세션 설정' }} />
      <Stack.Screen name="PlanPreview" component={PlanPreviewScreen} options={{ title: '플랜 미리보기' }} />
      <Stack.Screen
        name="ActiveSession"
        component={ActiveSessionScreen}
        options={{ title: '세션 진행 중', headerShown: false }}
      />
      <Stack.Screen
        name="SessionEnd"
        component={SessionEndScreen}
        options={{ title: '세션 완료', headerLeft: () => null }}
      />
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: '서버 설정' }} />
    </Stack.Navigator>
  );
}

function HistoryStack() {
  return (
    <Stack.Navigator screenOptions={DARK}>
      <Stack.Screen name="HistoryList" component={HistoryScreen} options={{ title: '훈련 기록' }} />
    </Stack.Navigator>
  );
}

function ProfileStack() {
  return (
    <Stack.Navigator screenOptions={DARK}>
      <Stack.Screen name="ProfileMain" component={ProfileScreen} options={{ title: '프로필' }} />
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: '서버 설정' }} />
    </Stack.Navigator>
  );
}

export default function App() {
  const loadServerUrl = useSettingsStore((s) => s.loadServerUrl);

  useEffect(() => {
    loadServerUrl();
    // Allow audio (including TTS) to play even when iPhone is on silent mode
    setAudioModeAsync({ playsInSilentMode: true });
  }, []);

  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={{
          headerShown: false,
          tabBarStyle: { backgroundColor: '#0a0a0a', borderTopColor: '#1a1a1a' },
          tabBarActiveTintColor: '#e63946',
          tabBarInactiveTintColor: '#555',
        }}
      >
        <Tab.Screen
          name="HomeTab"
          component={HomeStack}
          options={{ title: '홈', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>🥊</Text> }}
        />
        <Tab.Screen
          name="HistoryTab"
          component={HistoryStack}
          options={{ title: '기록', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>📋</Text> }}
        />
        <Tab.Screen
          name="ProfileTab"
          component={ProfileStack}
          options={{ title: '프로필', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>👤</Text> }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
