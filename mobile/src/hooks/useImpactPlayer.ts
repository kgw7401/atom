/**
 * useImpactPlayer — plays impact SFX as rhythm guides after coach voice.
 *
 * Pre-loads all impact sounds on mount, then plays action sequences
 * with timed gaps to simulate punching-bag hits.
 */

import { useCallback, useRef, useEffect } from 'react';
import { Audio } from 'expo-av';
import { IMPACT_ASSETS, CHAIN_ASSETS, CHAIN_TRIGGER_ACTIONS } from '../utils/sfxAssets';

// Timing constants (ms)
const INITIAL_DELAY = 300;   // delay after coach voice ends
const IMPACT_GAP = 250;      // gap between impact sounds

export interface ImpactPlayer {
  preload: () => Promise<void>;
  playSequence: (actions: string[]) => void;
  cancelPending: () => void;
  cleanup: () => Promise<void>;
}

export function useImpactPlayer(): ImpactPlayer {
  // Pre-loaded Sound instances: actionCode → Sound[]
  const soundsRef = useRef<Map<string, Audio.Sound[]>>(new Map());
  const chainSoundsRef = useRef<Audio.Sound[]>([]);

  // Variant rotation tracker: actionCode → last used index
  const lastVariantRef = useRef<Map<string, number>>(new Map());

  // Abort flag for pending sequences
  const abortRef = useRef(false);
  // Track active timeouts for cleanup
  const pendingTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const preload = useCallback(async () => {
    // Preload all impact sounds
    for (const [action, assets] of Object.entries(IMPACT_ASSETS)) {
      const sounds: Audio.Sound[] = [];
      for (const asset of assets) {
        try {
          const { sound } = await Audio.Sound.createAsync(asset, { shouldPlay: false });
          sounds.push(sound);
        } catch {
          // Skip failed loads gracefully
        }
      }
      if (sounds.length > 0) {
        soundsRef.current.set(action, sounds);
      }
    }

    // Preload chain rattle sounds
    for (const asset of CHAIN_ASSETS) {
      try {
        const { sound } = await Audio.Sound.createAsync(asset, { shouldPlay: false });
        chainSoundsRef.current.push(sound);
      } catch {}
    }
  }, []);

  const pickVariant = useCallback((action: string): Audio.Sound | null => {
    const sounds = soundsRef.current.get(action);
    if (!sounds || sounds.length === 0) return null;

    const lastIdx = lastVariantRef.current.get(action) ?? -1;
    // Pick next variant, avoiding repeat
    let nextIdx: number;
    if (sounds.length === 1) {
      nextIdx = 0;
    } else {
      nextIdx = (lastIdx + 1) % sounds.length;
    }
    lastVariantRef.current.set(action, nextIdx);
    return sounds[nextIdx];
  }, []);

  const playSound = useCallback(async (sound: Audio.Sound) => {
    try {
      await sound.setPositionAsync(0);
      await sound.playAsync();
    } catch {
      // Graceful failure — no crash on missing SFX
    }
  }, []);

  const playSequence = useCallback((actions: string[]) => {
    if (!actions || actions.length === 0) return;

    abortRef.current = false;

    // Clear any previous pending timers
    for (const t of pendingTimersRef.current) clearTimeout(t);
    pendingTimersRef.current = [];

    let delay = INITIAL_DELAY;

    for (let i = 0; i < actions.length; i++) {
      const action = actions[i];
      const currentDelay = delay;

      const timer = setTimeout(() => {
        if (abortRef.current) return;
        const sound = pickVariant(action);
        if (sound) playSound(sound);
      }, currentDelay);

      pendingTimersRef.current.push(timer);
      delay += IMPACT_GAP;
    }

    // Chain rattle after combos ending with hook/uppercut
    const lastAction = actions[actions.length - 1];
    if (CHAIN_TRIGGER_ACTIONS.has(lastAction) && chainSoundsRef.current.length > 0) {
      const chainDelay = delay + 50; // slight extra delay
      const chainSound = chainSoundsRef.current[
        Math.floor(Math.random() * chainSoundsRef.current.length)
      ];
      const timer = setTimeout(() => {
        if (abortRef.current) return;
        playSound(chainSound);
      }, chainDelay);
      pendingTimersRef.current.push(timer);
    }
  }, [pickVariant, playSound]);

  const cancelPending = useCallback(() => {
    abortRef.current = true;
    for (const t of pendingTimersRef.current) clearTimeout(t);
    pendingTimersRef.current = [];
  }, []);

  const cleanup = useCallback(async () => {
    cancelPending();
    // Unload all sounds
    for (const sounds of soundsRef.current.values()) {
      for (const sound of sounds) {
        try { await sound.unloadAsync(); } catch {}
      }
    }
    soundsRef.current.clear();
    for (const sound of chainSoundsRef.current) {
      try { await sound.unloadAsync(); } catch {}
    }
    chainSoundsRef.current = [];
  }, [cancelPending]);

  // Cleanup on unmount
  useEffect(() => {
    return () => { cleanup(); };
  }, [cleanup]);

  return { preload, playSequence, cancelPending, cleanup };
}
