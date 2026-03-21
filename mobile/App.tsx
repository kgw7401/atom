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
import SessionSetupScreen from './src/screens/SessionSetupScreen';
import PlanPreviewScreen from './src/screens/PlanPreviewScreen';
import ActiveSessionScreen from './src/screens/ActiveSessionScreen';
import SessionEndScreen from './src/screens/SessionEndScreen';
import FeedScreen from './src/screens/FeedScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import ExperienceScreen from './src/screens/onboarding/ExperienceScreen';
import EnvironmentScreen from './src/screens/onboarding/EnvironmentScreen';
import EquipmentScreen from './src/screens/onboarding/EquipmentScreen';
import GoalScreen from './src/screens/onboarding/GoalScreen';
import FrequencyScreen from './src/screens/onboarding/FrequencyScreen';

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
      <Stack.Screen name="SessionSetup" component={SessionSetupScreen} options={{ title: '세션 설정', headerBackTitle: '홈' }} />
      <Stack.Screen name="PlanPreview" component={PlanPreviewScreen} options={{ title: '플랜 미리보기' }} />
      <Stack.Screen
        name="ActiveSession"
        component={ActiveSessionScreen}
        options={{ headerShown: false, gestureEnabled: false }}
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
      <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: '서버 설정' }} />
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
        options={{ title: '홈', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>🥊</Text> }}
      />
      <Tab.Screen
        name="HistoryTab"
        component={HistoryStackScreen}
        options={{ title: '기록', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>📋</Text> }}
      />
      <Tab.Screen
        name="ProfileTab"
        component={ProfileStackScreen}
        options={{ title: '프로필', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 20 }}>👤</Text> }}
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
            name="Experience"
            component={ExperienceScreen}
            options={{ headerShown: false }}
          />
          <OnboardingStack.Screen
            name="Environment"
            component={EnvironmentScreen}
            options={{ title: '', headerBackTitle: '뒤로' }}
          />
          <OnboardingStack.Screen
            name="Equipment"
            component={EquipmentScreen}
            options={{ title: '', headerBackTitle: '뒤로' }}
          />
          <OnboardingStack.Screen
            name="Goal"
            component={GoalScreen}
            options={{ title: '', headerBackTitle: '뒤로' }}
          />
          <OnboardingStack.Screen
            name="Frequency"
            options={{ title: '', headerBackTitle: '뒤로' }}
          >
            {(props) => <FrequencyScreen {...props} onComplete={completeOnboarding} />}
          </OnboardingStack.Screen>
        </OnboardingStack.Navigator>
      )}
    </NavigationContainer>
  );
}
