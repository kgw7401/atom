import { useEffect, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = 'atom_onboarding_complete';

export function useOnboarding() {
  const [isLoading, setIsLoading] = useState(true);
  const [isOnboarded, setIsOnboarded] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(KEY).then((value) => {
      setIsOnboarded(value === 'true');
      setIsLoading(false);
    });
  }, []);

  const completeOnboarding = async () => {
    await AsyncStorage.setItem(KEY, 'true');
    setIsOnboarded(true);
  };

  return { isLoading, isOnboarded, completeOnboarding };
}
