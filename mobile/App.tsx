import React, { useEffect } from 'react';
import { setAudioModeAsync } from 'expo-audio';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { Text, ActivityIndicator, View } from 'react-native';

// Global error handler — logs full error to Metro terminal
const _ErrorUtils = (global as any).ErrorUtils;
if (_ErrorUtils) {
  const defaultHandler = _ErrorUtils.getGlobalHandler();
  _ErrorUtils.setGlobalHandler((error: any, isFatal: boolean) => {
    console.error('[ATOM ERROR]', isFatal ? 'FATAL' : 'NON-FATAL', error?.message);
    console.error(error?.stack);
    defaultHandler(error, isFatal);
  });
}

import { useSettingsStore } from './src/store/settingsStore';
import { useOnboarding } from './src/hooks/useOnboarding';
import { COLORS } from './src/theme';

import HomeScreen from './src/screens/HomeScreen';
import ActiveSessionScreen from './src/screens/ActiveSessionScreen';
import SessionEndScreen from './src/screens/SessionEndScreen';
import FeedScreen from './src/screens/FeedScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import OnboardingWelcomeScreen from './src/screens/onboarding/OnboardingWelcomeScreen';
import OnboardingExperienceScreen from './src/screens/onboarding/OnboardingExperienceScreen';
import OnboardingPreferenceScreen from './src/screens/onboarding/OnboardingPreferenceScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();
const OnboardingStack = createNativeStackNavigator();

const DARK = {
  headerStyle: { backgroundColor: COLORS.BG },
  headerTintColor: '#fff',
  headerTitleStyle: { fontWeight: '700' as const },
  contentStyle: { backgroundColor: COLORS.BG },
};

function HomeStackScreen() {
  return (
    <Stack.Navigator screenOptions={DARK}>
      <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
      <Stack.Screen
        name="ActiveSession"
        component={ActiveSessionScreen}
        options={{ headerShown: false, gestureEnabled: false }}
      />
      <Stack.Screen
        name="SessionEnd"
        component={SessionEndScreen}
        options={{ headerShown: false, gestureEnabled: false }}
      />
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: '설정' }} />
    </Stack.Navigator>
  );
}

function HistoryStackScreen() {
  return (
    <Stack.Navigator screenOptions={DARK}>
      <Stack.Screen name="Feed" component={FeedScreen} options={{ title: '훈련 기록' }} />
    </Stack.Navigator>
  );
}

function ProfileStackScreen() {
  return (
    <Stack.Navigator screenOptions={DARK}>
      <Stack.Screen name="ProfileMain" component={ProfileScreen} options={{ title: '프로필' }} />
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: '설정' }} />
    </Stack.Navigator>
  );
}

function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: { backgroundColor: COLORS.BG, borderTopColor: COLORS.SURFACE },
        tabBarActiveTintColor: COLORS.RED,
        tabBarInactiveTintColor: COLORS.TEXT_3,
      }}
    >
      <Tab.Screen
        name="HomeTab"
        component={HomeStackScreen}
        options={{ title: '훈련', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>🥊</Text> }}
      />
      <Tab.Screen
        name="HistoryTab"
        component={HistoryStackScreen}
        options={{ title: '기록', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>📋</Text> }}
      />
      <Tab.Screen
        name="ProfileTab"
        component={ProfileStackScreen}
        options={{ title: '나', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>👤</Text> }}
      />
    </Tab.Navigator>
  );
}

export default function App() {
  const loadServerUrl = useSettingsStore((s) => s.loadServerUrl);
  const { isLoading, isOnboarded, completeOnboarding } = useOnboarding();

  useEffect(() => {
    loadServerUrl();
    setAudioModeAsync({ playsInSilentMode: true });
  }, []);

  if (isLoading) {
    return (
      <View style={{ flex: 1, backgroundColor: COLORS.BG, alignItems: 'center', justifyContent: 'center' }}>
        <ActivityIndicator color={COLORS.RED} />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <StatusBar style="light" />
      {isOnboarded ? (
        <TabNavigator />
      ) : (
        <OnboardingStack.Navigator screenOptions={{ ...DARK, animation: 'slide_from_right' }}>
          <OnboardingStack.Screen
            name="Welcome"
            component={OnboardingWelcomeScreen}
            options={{ headerShown: false }}
          />
          <OnboardingStack.Screen
            name="Experience"
            component={OnboardingExperienceScreen}
            options={{ headerShown: false }}
          />
          <OnboardingStack.Screen
            name="Preference"
            options={{ headerShown: false }}
          >
            {(props) => <OnboardingPreferenceScreen {...props} onComplete={completeOnboarding} />}
          </OnboardingStack.Screen>
        </OnboardingStack.Navigator>
      )}
    </NavigationContainer>
  );
}
