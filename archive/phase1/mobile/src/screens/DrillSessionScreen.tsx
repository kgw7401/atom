import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { Audio } from 'expo-av';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'DrillSession'>;

export default function DrillSessionScreen({ route, navigation }: Props) {
  const { sessionId, script } = route.params;
  const [hasPermission, setHasPermission] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentInstruction, setCurrentInstruction] = useState(0);
  const [countdown, setCountdown] = useState(3);
  const [sessionTime, setSessionTime] = useState(0);

  const cameraRef = useRef<CameraView>(null);
  const recordingRef = useRef<any>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const instructionTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    requestPermissions();
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (instructionTimerRef.current) clearTimeout(instructionTimerRef.current);
      stopRecording();
    };
  }, []);

  // Auto-start countdown when permissions are granted
  useEffect(() => {
    if (hasPermission && countdown > 0 && countdown <= 3) {
      const timer = setTimeout(() => {
        setCountdown((prev) => prev - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [hasPermission, countdown]);

  // When countdown reaches 0, wait a bit then start session (fallback for simulator)
  useEffect(() => {
    if (countdown === 0 && hasPermission && !isRecording) {
      const timer = setTimeout(() => {
        // If still not recording after 2 seconds, force start
        if (!isRecording) {
          startSession();
        }
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [countdown, hasPermission, isRecording]);

  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [microphonePermission, requestMicrophonePermission] = useMicrophonePermissions();

  const requestPermissions = async () => {
    const camera = await requestCameraPermission();
    const audio = await Audio.requestPermissionsAsync();

    if (camera.granted && audio.status === 'granted') {
      setHasPermission(true);
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
    } else {
      Alert.alert('Permission Required', 'Camera and microphone permissions are required');
      navigation.goBack();
    }
  };


  const startSession = async () => {
    // Start recording
    if (cameraRef.current) {
      try {
        // Start recording (don't await - let it run in background)
        cameraRef.current.recordAsync({
          maxDuration: script.estimated_duration_seconds + 10,
        }).then((video) => {
          recordingRef.current = video;
          finishSession();
        }).catch((error) => {
          console.error('Recording failed on simulator:', error);
          // Simulator doesn't support recording, finish without video
          setTimeout(() => {
            finishSessionWithoutVideo();
          }, script.estimated_duration_seconds * 1000);
        });

        setIsRecording(true);

        // Start session timer
        timerRef.current = setInterval(() => {
          setSessionTime((t) => t + 0.1);
        }, 100);

        // Schedule audio instructions
        scheduleInstructions();
      } catch (error) {
        console.error('Recording error:', error);
        Alert.alert('Error', 'Failed to start recording. Continuing without video...');
        // Continue without recording for testing
        startWithoutRecording();
      }
    } else {
      startWithoutRecording();
    }
  };

  const startWithoutRecording = () => {
    setIsRecording(true);

    // Start session timer
    timerRef.current = setInterval(() => {
      setSessionTime((t) => t + 0.1);
    }, 100);

    // Schedule audio instructions
    scheduleInstructions();

    // Auto-finish after duration (since no recording to trigger it)
    setTimeout(() => {
      finishSessionWithoutVideo();
    }, script.estimated_duration_seconds * 1000);
  };

  const finishSessionWithoutVideo = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (instructionTimerRef.current) clearTimeout(instructionTimerRef.current);

    Alert.alert(
      'Session Complete',
      'Testing mode: No video was recorded. In production, video would be uploaded here.',
      [
        {
          text: 'Go Back',
          onPress: () => navigation.goBack(),
        },
      ]
    );
  };

  const scheduleInstructions = () => {
    script.instructions.forEach((instruction, index) => {
      const delay = instruction.t * 1000;
      setTimeout(() => {
        setCurrentInstruction(index);
        // Play audio - in production, load pre-recorded audio files
        // For MVP, just show text
      }, delay);
    });
    // Note: Auto-finish is handled by each recording path
    // - startWithoutRecording() calls finishSessionWithoutVideo()
    // - recordAsync() maxDuration handles recording finish
  };

  const stopRecording = async () => {
    if (cameraRef.current && isRecording) {
      await cameraRef.current.stopRecording();
      setIsRecording(false);
    }
  };

  const finishSession = async () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (instructionTimerRef.current) clearTimeout(instructionTimerRef.current);

    await stopRecording();

    if (recordingRef.current) {
      navigation.navigate('Uploading', {
        sessionId,
        videoUri: recordingRef.current.uri,
      });
    } else {
      Alert.alert('Error', 'No recording found');
      navigation.goBack();
    }
  };

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Requesting permissions...</Text>
      </View>
    );
  }

  if (countdown > 0) {
    return (
      <View style={styles.container}>
        <Text style={styles.countdownText}>{countdown}</Text>
        <Text style={styles.text}>Get ready!</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="front"
      />

      {/* Instruction Overlay */}
      <View style={styles.overlay}>
        <View style={styles.instructionBox}>
          <Text style={styles.instructionText}>
            {script.instructions[currentInstruction]?.display || 'Get ready...'}
          </Text>
          <Text style={styles.timerText}>{sessionTime.toFixed(1)}s</Text>
        </View>

        {isRecording && (
          <TouchableOpacity style={styles.finishButton} onPress={finishSession}>
            <Text style={styles.finishButtonText}>Finish Early</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    padding: 20,
  },
  instructionBox: {
    backgroundColor: 'rgba(231, 76, 60, 0.9)',
    padding: 30,
    borderRadius: 15,
    alignItems: 'center',
    marginTop: 60,
  },
  instructionText: {
    color: '#fff',
    fontSize: 32,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  timerText: {
    color: '#fff',
    fontSize: 18,
    marginTop: 10,
  },
  countdownText: {
    color: '#e74c3c',
    fontSize: 120,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  text: {
    color: '#fff',
    fontSize: 24,
    textAlign: 'center',
    marginTop: 20,
  },
  finishButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    padding: 15,
    borderRadius: 10,
    alignSelf: 'center',
    marginBottom: 40,
  },
  finishButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
