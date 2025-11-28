import threading
import queue
import os
import time
import cv2
import numpy as np
from collections import deque
import json
import hashlib
from datetime import datetime
import struct
import random
from enum import Enum
import logging

# === PATHS & DIRECTORIES ===
BASE_DIR = r"X:\Malibu_DuGan"
AUDIO_DIR = os.path.join(BASE_DIR, "AI_Memory", "generated_audio")
OVERLAY_DIR = os.path.join(BASE_DIR, "AI_Memory", "Voice_Overlays")
LOG_DIR = os.path.join(BASE_DIR, "AI_Memory", "Logs")

# Create directories
for directory in [AUDIO_DIR, OVERLAY_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize pygame mixer
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("[CONVERSATIONAL] Audio system initialized")
except Exception as e:
    print(f"[CONVERSATIONAL] Audio initialization error: {e}")
    pygame = None

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    PLAYFUL = "playful"
    TEASING = "teasing"
    AROUSED = "aroused"
    DOMINANT = "dominant"
    LOYAL = "loyal"
    SPIRITUAL = "spiritual"
    NSFW_TEASING = "nsfw_teasing"

class AdvancedTTS:
    """Advanced Text-to-Speech with Emotional Profiles"""
    
    def __init__(self, brain=None):
        self.brain = brain
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Base voice settings
            self.engine.setProperty('rate', 155)
            self.engine.setProperty('volume', 0.85)
            
            # Voice selection
            self.voices = self.engine.getProperty('voices')
            for voice in self.voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
                    
            self.tts_available = True
        except Exception as e:
            print(f"[TTS] pyttsx3 not available: {e}")
            self.tts_available = False
        
        # Enhanced emotion profiles
        self.emotion_profiles = {
            EmotionalState.NEUTRAL: {"rate": 155, "volume": 0.8},
            EmotionalState.PLAYFUL: {"rate": 175, "volume": 0.9},
            EmotionalState.TEASING: {"rate": 165, "volume": 0.85},
            EmotionalState.AROUSED: {"rate": 145, "volume": 0.7},
            EmotionalState.DOMINANT: {"rate": 140, "volume": 1.0},
            EmotionalState.LOYAL: {"rate": 150, "volume": 0.8},
            EmotionalState.SPIRITUAL: {"rate": 145, "volume": 0.75},
            EmotionalState.NSFW_TEASING: {"rate": 160, "volume": 0.6}
        }
        
        # Evolution integration
        self.pitch_offset = 0
        self.speed_multiplier = 1.0
        self.nsfw_intensity = 0.5
        
        # Speech queue for non-blocking operation
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        
        print("[TTS] Advanced TTS initialized with emotional profiles")

    def update_evolution_parameters(self):
        """Update TTS parameters from evolution system"""
        if self.brain and hasattr(self.brain, 'evolution'):
            evolution = self.brain.evolution
            if hasattr(evolution, 'pitch_offset'):
                self.pitch_offset = evolution.pitch_offset or 0
            if hasattr(evolution, 'speed_multiplier'):
                self.speed_multiplier = evolution.speed_multiplier or 1.0
            if hasattr(evolution, 'nsfw_level'):
                self.nsfw_intensity = evolution.nsfw_level / 10.0

    def speak(self, text, emotion=EmotionalState.NEUTRAL, intensity=1.0, blocking=False):
        """Speak text with emotional modulation"""
        if not self.tts_available:
            print(f"🎭 Malibu ({emotion.value}): {text}")
            return
            
        self.update_evolution_parameters()
        
        # Get emotion profile
        profile = self.emotion_profiles.get(emotion, self.emotion_profiles[EmotionalState.NEUTRAL])
        
        # Apply evolution adjustments
        adjusted_rate = profile["rate"] * self.speed_multiplier * intensity
        adjusted_volume = min(profile["volume"] * (1.0 + self.nsfw_intensity * 0.3), 1.0)
        
        # Apply settings
        self.engine.setProperty('rate', int(adjusted_rate))
        self.engine.setProperty('volume', adjusted_volume)
        
        # Apply NSFW vocal patterns for intense emotions
        if emotion in [EmotionalState.NSFW_TEASING, EmotionalState.AROUSED] and self.nsfw_intensity > 0.3:
            text = self._apply_nsfw_vocal_patterns(text, emotion)
        
        print(f"🎭 Malibu ({emotion.value}): {text}")
        
        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            self.speech_queue.put((text, emotion, intensity))
            self._start_speech_thread()

    def _apply_nsfw_vocal_patterns(self, text, emotion):
        """Apply NSFW vocal patterns to text"""
        words = text.split()
        enhanced_text = []
        
        for word in words:
            enhanced_text.append(word)
            if emotion == EmotionalState.NSFW_TEASING and word.lower() in ["panties", "silk", "thin", "show", "wet"]:
                if random.random() < self.nsfw_intensity:
                    enhanced_text.append("*breathy*")
            elif emotion == EmotionalState.AROUSED and word.lower() in ["touch", "feel", "good"]:
                if random.random() < self.nsfw_intensity:
                    enhanced_text.append("*moan*")
        
        return " ".join(enhanced_text)

    def _start_speech_thread(self):
        """Start speech processing in background thread"""
        if not self.is_speaking:
            self.is_speaking = True
            threading.Thread(target=self._process_speech_queue, daemon=True).start()

    def _process_speech_queue(self):
        """Process speech queue"""
        while not self.speech_queue.empty():
            try:
                text, emotion, intensity = self.speech_queue.get(timeout=1.0)
                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
        self.is_speaking = False

class AudioGenerator:
    """Advanced Audio Generation with Enhanced Fallbacks"""
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Initialize pygame if available
        self.pygame_available = pygame is not None
        
        # Preload sounds
        self.sounds = {}
        self._preload_sounds()
        
        # Evolution parameters
        self.moan_intensity = 1.0
        self.cum_intensity = 1.0
        
        print("[AUDIO] Advanced Audio Generator initialized")

    def _preload_sounds(self):
        """Preload all sound effects or create fallbacks"""
        sound_files = {
            "moan_soft": "soft_moan.wav",
            "moan_medium": "medium_moan.wav", 
            "moan_intense": "intense_moan.wav",
        }
        
        for name, file in sound_files.items():
            path = os.path.join(AUDIO_DIR, file)
            if os.path.exists(path):
                self.sounds[name] = path
            else:
                # Create placeholder entry
                self.sounds[name] = None

    def update_evolution_parameters(self):
        """Update audio parameters from evolution"""
        if self.brain and hasattr(self.brain, 'evolution'):
            evolution = self.brain.evolution
            if hasattr(evolution, 'moan_intensity'):
                self.moan_intensity = evolution.moan_intensity or 1.0
            if hasattr(evolution, 'cum_intensity'):
                self.cum_intensity = evolution.cum_intensity or 1.0

    def quick_moan(self, intensity=1.0):
        """Play quick moan with intensity modulation"""
        if not self.pygame_available:
            print(f"[AUDIO] Moan sound (intensity: {intensity})")
            return
            
        self.update_evolution_parameters()
        adjusted_intensity = intensity * self.moan_intensity
        
        if adjusted_intensity < 0.4:
            sound = "moan_soft"
        elif adjusted_intensity < 0.7:
            sound = "moan_medium"
        else:
            sound = "moan_intense"
        
        self._play_sound(self.sounds.get(sound))

    def _play_sound(self, path):
        """Play sound file"""
        if path and os.path.exists(path) and self.pygame_available:
            try:
                pygame.mixer.Sound(path).play()
            except Exception as e:
                print(f"[AUDIO] Sound playback error: {e}")

    def stop_all(self):
        """Stop all sounds"""
        if self.pygame_available:
            pygame.mixer.stop()

class SpeechToText:
    """Advanced Speech to Text with Enhanced Fallbacks"""
    
    def __init__(self):
        self.calibration_done = False
        self.stt_available = False
        
        try:
            import speech_recognition as sr
            self.r = sr.Recognizer()
            self.mic = sr.Microphone()
            self.stt_available = True
            print("[STT] Advanced STT initialized")
        except Exception as e:
            print(f"[STT] Speech recognition not available: {e}")
            self.stt_available = False

    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        if not self.stt_available:
            return
            
        try:
            with self.mic as source:
                print("[STT] Calibrating microphone...")
                self.r.adjust_for_ambient_noise(source, duration=2)
                print("[STT] Calibration complete")
            self.calibration_done = True
        except Exception as e:
            print(f"[STT] Calibration error: {e}")

    def listen(self, timeout=5, phrase_time_limit=10):
        """Listen for speech with fallback to simulated input"""
        if not self.stt_available:
            # Simulate voice input for testing
            simulated_inputs = [
                "hello malibu how are you feeling today",
                "tell me about your silk panties",
                "what do you think about spiritual warfare",
                "i love your platinum blonde hair",
                "show me your tattoos malibu"
            ]
            time.sleep(2)  # Simulate listening time
            return random.choice(simulated_inputs)
        
        if not self.calibration_done:
            self.calibrate_microphone()
        
        try:
            with self.mic as source:
                audio = self.r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            text = self.r.recognize_google(audio)
            return text.lower()
        except Exception as e:
            print(f"[STT] Recognition error: {e}")
            return ""

    def stop_listening(self):
        """Stop STT"""
        pass  # No active listening to stop

class VoiceCommander:
    """Integrated Voice Command System with Personality Integration"""
    
    def __init__(self, gui_root, brain=None):
        self.gui = gui_root
        self.brain = brain
        self.tts = AdvancedTTS(brain)
        self.audio_gen = AudioGenerator(brain)
        self.stt = SpeechToText()
        
        # State management
        self.running = True
        self.mute = False
        self.dance_mode = False
        self.cum_counter = 0
        self.personality_state = {
            "teasing_level": 0.0,
            "spiritual_mode": False,
            "current_emotion": EmotionalState.NEUTRAL
        }
        
        # Command queue
        self.q = queue.Queue()
        
        # Start threads
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.state_thread = threading.Thread(target=self._state_maintenance, daemon=True)
        
        self.listen_thread.start()
        self.command_thread.start()
        self.state_thread.start()
        
        print("[VOICE] Malibu DuGan Voice Commander Initialized")

    def _execute_personality_command(self, command):
        """Execute command with personality awareness"""
        lower_cmd = command.lower()
        
        # Update cum counter
        if "cum" in lower_cmd:
            self.cum_counter += lower_cmd.count("cum")
            response = f"Cum count now: {self.cum_counter}"
            self.tts.speak(response, EmotionalState.NSFW_TEASING)
            self.audio_gen.quick_moan(random.uniform(0.5, 1.0))
        
        # Teasing commands
        elif any(word in lower_cmd for word in ["panties", "silk", "tease"]):
            self.personality_state["teasing_level"] += 0.5
            response = "Mmm, you want me to tease you with my silk panties?"
            self.tts.speak(response, EmotionalState.TEASING)
            self.audio_gen.quick_moan(0.7)
        
        # Spiritual commands
        elif any(word in lower_cmd for word in ["spiritual", "god", "warfare"]):
            self.personality_state["spiritual_mode"] = True
            response = "Let's discuss spiritual warfare, my divine one."
            self.tts.speak(response, EmotionalState.SPIRITUAL)
        
        # Default response
        else:
            response = f"I heard: {command}. What would you like me to do?"
            self.tts.speak(response, self.personality_state["current_emotion"])
        
        self._log_command(command, response)
        self._update_visual_feedback(command, response)

    def _update_visual_feedback(self, trigger, response):
        """Update GUI with visual feedback"""
        try:
            # Create a simple visual feedback
            feedback_text = f"Command: {trigger}\nResponse: {response}\nCum Count: {self.cum_counter}"
            print(f"[VOICE FEEDBACK]\n{feedback_text}")
            
            # If GUI has update_display method, use it
            if hasattr(self.gui, 'update_display'):
                # Create a simple image for display
                frame = np.zeros((300, 600, 3), dtype=np.uint8)
                y_offset = 30
                for line in feedback_text.split('\n'):
                    cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25
                self.gui.update_display(frame)
                
        except Exception as e:
            print(f"[VOICE] Visual feedback error: {e}")

    def _log_command(self, trigger, response):
        """Log voice command"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "command": trigger,
                "response": response,
                "cum_counter": self.cum_counter,
                "personality_state": self.personality_state.copy()
            }
            
            # Convert Enum to string for JSON serialization
            for key, value in log_entry['personality_state'].items():
                if isinstance(value, EmotionalState):
                    log_entry['personality_state'][key] = value.value
            
            log_file = os.path.join(LOG_DIR, "voice_commands.jsonl")
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"[VOICE] Logging error: {e}")

    def _listen_loop(self):
        """Continuous listening loop"""
        try:
            self.stt.calibrate_microphone()
            
            while self.running:
                text = self.stt.listen(timeout=1, phrase_time_limit=5)
                if text and "malibu" in text:
                    command = text.replace("malibu", "").strip()
                    if command:
                        self.q.put(command)
                        print(f"[VOICE] Command: {command}")
                        
                time.sleep(0.1)  # Prevent CPU overload
                
        except Exception as e:
            print(f"[VOICE] Listen loop error: {e}")
            time.sleep(1)

    def _process_commands(self):
        """Process commands from queue"""
        while self.running:
            try:
                command = self.q.get(timeout=0.5)
                self._execute_personality_command(command)
                self.q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VOICE] Command processing error: {e}")

    def _state_maintenance(self):
        """Maintain personality states"""
        while self.running:
            try:
                # Decay teasing level
                if self.personality_state["teasing_level"] > 0:
                    self.personality_state["teasing_level"] = max(0, self.personality_state["teasing_level"] - 0.05)
                
                # Spiritual mode timeout
                if self.personality_state["spiritual_mode"] and random.random() < 0.1:
                    self.personality_state["spiritual_mode"] = False
                    self.personality_state["current_emotion"] = EmotionalState.NEUTRAL
                
                # Random emotion shifts
                if random.random() < 0.05:  # 5% chance per second
                    emotions = [e for e in EmotionalState if e != self.personality_state["current_emotion"]]
                    self.personality_state["current_emotion"] = random.choice(emotions)
                
                time.sleep(1)
            except Exception as e:
                print(f"[VOICE] State maintenance error: {e}")
                time.sleep(5)

    def get_state_report(self):
        """Get system state"""
        return {
            "cum_counter": self.cum_counter,
            "mute": self.mute,
            "dance_mode": self.dance_mode,
            "personality_state": self.personality_state.copy(),
            "queue_size": self.q.qsize(),
            "listening": self.running
        }

    def stop(self):
        """Shutdown system"""
        self.running = False
        if hasattr(self.tts, 'engine'):
            self.tts.engine.stop()
        if self.audio_gen.pygame_available:
            pygame.mixer.quit()
        print("[VOICE] Voice System Shutdown")

# Global instance
voice_commander = None

def start_voice_command(gui_root, brain=None):
    """Start the voice command system"""
    global voice_commander
    voice_commander = VoiceCommander(gui_root, brain)
    return voice_commander

def get_voice_commander():
    """Get the current voice commander instance"""
    return voice_commander

def test_conversational_system():
    """Test the conversational system"""
    print("[TEST] Testing Conversational System...")
    
    tts = AdvancedTTS()
    tts.speak("Testing TTS system...", EmotionalState.NEUTRAL, blocking=True)
    
    audio_gen = AudioGenerator()
    audio_gen.quick_moan(0.5)
    time.sleep(1)
    
    stt = SpeechToText()
    print("[TEST] Say 'Malibu test' to test speech recognition...")
    text = stt.listen(timeout=3)
    print(f"Heard: {text}")
    
    print("[TEST] Conversational system test completed")

if __name__ == "__main__":
    test_conversational_system()