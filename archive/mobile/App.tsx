import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import type { RootStackParamList } from './src/navigation/types';

import HomeScreen from './src/screens/HomeScreen';
import SessionSetupScreen from './src/screens/SessionSetupScreen';
import DrillSessionScreen from './src/screens/DrillSessionScreen';
import UploadingScreen from './src/screens/UploadingScreen';
import ReportScreen from './src/screens/ReportScreen';
import TwinScreen from './src/screens/TwinScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            backgroundColor: '#000',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
          contentStyle: {
            backgroundColor: '#000',
          },
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="SessionSetup"
          component={SessionSetupScreen}
          options={{ title: 'Setup Session' }}
        />
        <Stack.Screen
          name="DrillSession"
          component={DrillSessionScreen}
          options={{
            headerShown: false,
            gestureEnabled: false,
          }}
        />
        <Stack.Screen
          name="Uploading"
          component={UploadingScreen}
          options={{
            title: 'Processing',
            headerShown: false,
            gestureEnabled: false,
          }}
        />
        <Stack.Screen
          name="Report"
          component={ReportScreen}
          options={{
            title: 'Session Report',
            headerLeft: () => null,
            gestureEnabled: false,
          }}
        />
        <Stack.Screen
          name="Twin"
          component={TwinScreen}
          options={{ title: 'Digital Twin' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
