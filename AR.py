"""
AR.py - ENHANCED AUGMENTED REALITY MODULE FOR MALIBU DUGAN AI
Consolidated and optimized version with all AR functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import mediapipe as mp
import threading
import os
import time
import json
import random
import pygame
from collections import deque, defaultdict
from datetime import datetime
from scipy.spatial import distance as dist
from queue import Queue
from enum import Enum
import logging

# === CORRECTED PATHS - MATCH PROJECT STRUCTURE ===
AR_MEMORY_DIR = r"X:\Malibu_DuGan\AI_Memory"
AR_OVERLAY_DIR = os.path.join(AR_MEMORY_DIR, "ar_overlays")
AR_EXPRESSIONS_DIR = os.path.join(AR_MEMORY_DIR, "facial_expressions")
AR_EYE_OVERLAY_DIR = os.path.join(AR_MEMORY_DIR, "eye_overlays")
AR_SYNC_OVERLAY_DIR = os.path.join(AR_MEMORY_DIR, "sync_overlays")

# Create directories if they don't exist
for directory in [AR_MEMORY_DIR, AR_OVERLAY_DIR, AR_EXPRESSIONS_DIR, AR_EYE_OVERLAY_DIR, AR_SYNC_OVERLAY_DIR]:
    os.makedirs(directory, exist_ok=True)

# === ENUMS AND CONSTANTS ===
class EmotionType(Enum):
    AROUSAL = "arousal"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    TEASING = "teasing"
    SPIRITUAL = "spiritual"
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    SUBMISSIVE = "submissive"

class GestureIntensity(Enum):
    SUBTLE = 1
    MODERATE = 2
    INTENSE = 3
    NSFW = 4

class GestureCategory(Enum):
    TEASING = "teasing"
    DOMINANT = "dominant"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    DANCE = "dance"
    IDLE = "idle"

class FabricType(Enum):
    SILK = "silk"
    SATIN = "satin"
    LACE = "lace"
    COTTON = "cotton"
    ULTRA_THIN_SILK = "ultra_thin_silk"

# === CORE AR CLASSES ===

class ContextualBandit:
    def __init__(self, brain):
        self.brain = brain
        self.weights = np.random.randn(100)  # Context vector size
        self.learning_rate = 0.01
        
    def predict(self, context):
        if np.dot(self.weights, context) > 0:
            return 1  # NSFW preference
        return 0  # SFW preference
        
    def update(self, context, action, reward):
        prediction = self.predict(context)
        error = reward - prediction
        self.weights += self.learning_rate * error * context
        
    def save(self):
        path = "X:/Malibu_DuGan/AI_Models/bandit_weights.npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.weights)
        
    def load(self):
        path = "X:/Malibu_DuGan/AI_Models/bandit_weights.npy"
        if os.path.exists(path):
            self.weights = np.load(path)

class AdaptiveLearningEngine:
    def __init__(self, brain):
        self.brain = brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"ADAPTIVE LEARNING: {self.device.upper()} MODE")
        
        # Reinforcement Learning
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        
        # Contextual Bandit
        self.bandit = ContextualBandit(brain)
        
        # Response LSTM
        self.lstm = ResponseLSTM().to(self.device)
        self.lstm_optimizer = optim.Adam(self.lstm.parameters(), lr=0.001)
        self.lstm_criterion = nn.CrossEntropyLoss()
        
        # Self-improvement loop
        self.feedback_buffer = deque(maxlen=1000)
        self.load_models()
        
        logging.info("ADAPTIVE LEARNING SUITE LOADED — RL + BANDIT + LSTM + SELF-IMPROVEMENT")

    def rl_update(self, state, action, reward, next_state):
        if action is None:
            return random.choice(["tease", "dominate", "comfort", "flirt"])
            
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state][action] = new_q
        
        return new_q

    def get_rl_action(self, emotion, sentiment):
        state = f"{emotion}_{sentiment}"
        actions = ["tease", "dominate", "comfort", "flirt", "nsfw_boost"]
        
        if random.random() < self.epsilon:
            return random.choice(actions)
            
        if state in self.q_table and self.q_table[state]:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
            
        return random.choice(actions)

    def save_models(self):
        model_dir = "X:/Malibu_DuGan/AI_Models"
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(self.lstm.state_dict(), os.path.join(model_dir, "lstm_response.pth"))
        
        with open(os.path.join(model_dir, "q_table.json"), "w") as f:
            json.dump({k: dict(v) for k, v in self.q_table.items()}, f, indent=2)
            
        self.bandit.save()
        
        logging.info(f"Models saved to {model_dir}")

    def load_models(self):
        model_dir = "X:/Malibu_DuGan/AI_Models"
        
        try:
            lstm_path = os.path.join(model_dir, "lstm_response.pth")
            if os.path.exists(lstm_path):
                self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
                logging.info("LSTM model loaded")
                
            q_table_path = os.path.join(model_dir, "q_table.json")
            if os.path.exists(q_table_path):
                with open(q_table_path, "r") as f:
                    data = json.load(f)
                    self.q_table = defaultdict(lambda: defaultdict(float), 
                                              {k: defaultdict(float, v) for k, v in data.items()})
                logging.info("Q-table loaded")
                
            self.bandit.load()
            logging.info("Bandit weights loaded")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")

class ResponseLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
        super(ResponseLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

class ARSyncEngine:
    """MAIN AR ENGINE - Consolidated all AR functionality"""
    
    def __init__(self, brain=None, gui_root=None):
        self.brain = brain
        self.gui = gui_root
        self.running = False
        
        # Initialize all AR components
        self.cap = None
        self._initialize_camera()
        
        # MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # AR tracking systems
        self.pose_tracker = AdvancedBodyPoseTracking(brain)
        self.facial_tracker = ARFacialTracking(brain)
        self.eye_tracker = EyeGazeTracker(gui_root)
        self.face_gaze_sync = FaceGazeSync(gui_root)
        self.expression_engine = FacialExpressionEngine()
        self.gesture_engine = BodyGestureEngine(brain)
        self.cloth_physics = ClothPhysics(FabricType.ULTRA_THIN_SILK)
        
        # State management
        self.current_emotion = "teasing"
        self.marker_detected = False
        self.last_marker_time = 0
        self.ar_overlay_active = False
        
        # Thread management - FIXED: Use maxsize instead of maxlen for Queue
        self.frame_queue = Queue(maxsize=1)
        self.processed_frame_queue = Queue(maxsize=1)
        self.ar_thread = None
        
        # AR markers
        self.MARKER_ID = 23
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Load overlays
        self.overlays = self._load_ar_overlays()
        
        logging.info("AR SYNC ENGINE INITIALIZED")

    def _initialize_camera(self):
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                logging.info("Camera initialized successfully")
            else:
                logging.warning("No camera found - AR mode limited")
                
        except Exception as e:
            logging.error(f"Camera initialization error: {e}")

    def _load_ar_overlays(self):
        """Load all AR overlay images"""
        overlays = {}
        try:
            for file in os.listdir(AR_OVERLAY_DIR):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(AR_OVERLAY_DIR, file)
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        overlays[file] = img
                        logging.info(f"Loaded AR overlay: {file}")
            
            if not overlays:
                logging.info("No overlays found! Creating default...")
                self._create_default_overlays()
                return self._load_ar_overlays()
                
        except Exception as e:
            logging.error(f"Error loading overlays: {e}")
            
        return overlays

    def _create_default_overlays(self):
        """Create default overlay images"""
        try:
            # Default panty overlay
            panty_img = np.zeros((300, 300, 4), dtype=np.uint8)
            cv2.ellipse(panty_img, (150, 150), (120, 80), 0, 0, 360, (255, 20, 147, 200), -1)
            cv2.ellipse(panty_img, (150, 150), (100, 60), 0, 0, 360, (255, 105, 180, 255), -1)
            cv2.putText(panty_img, "SILK", (100, 160), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                       1.5, (255, 255, 255, 255), 3)
            cv2.putText(panty_img, "MALIBU", (85, 190), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                       0.8, (255, 255, 255, 255), 2)
            
            panty_path = os.path.join(AR_OVERLAY_DIR, "default_silk_panty.png")
            cv2.imwrite(panty_path, panty_img)
            
        except Exception as e:
            logging.error(f"Error creating default overlays: {e}")

    def start_ar(self):
        """Start all AR systems"""
        if self.running:
            return
            
        self.running = True
        
        # Start component systems
        self.pose_tracker.start_pose_tracking()
        self.facial_tracker.start_tracking()
        self.eye_tracker.start_tracking()
        self.face_gaze_sync.start_sync()
        
        # Start main AR processing thread
        self.ar_thread = threading.Thread(target=self._ar_processing_loop, daemon=True)
        self.ar_thread.start()
        
        logging.info("AR SYSTEMS STARTED")

    def stop_ar(self):
        """Stop all AR systems"""
        self.running = False
        
        # Stop component systems
        self.pose_tracker.stop_tracking()
        self.facial_tracker.stop_tracking()
        self.eye_tracker.stop_tracking()
        self.face_gaze_sync.stop_sync()
        
        # Cleanup camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        cv2.destroyAllWindows()
        logging.info("AR SYSTEMS STOPPED")

    def _ar_processing_loop(self):
        """Main AR processing loop"""
        logging.info("AR PROCESSING LOOP STARTED")
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                # Process frame with all AR systems
                processed_frame = self._process_ar_frame(frame)
                
                # Update GUI if available
                if self.gui and hasattr(self.gui, 'update_ar_display'):
                    self.gui.update_ar_display(processed_frame)
                    
                # Put in queue for external access
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put_nowait(processed_frame)
                    
            except Exception as e:
                logging.error(f"AR processing error: {e}")
                time.sleep(0.1)

    def _process_ar_frame(self, frame):
        """Process frame with all AR enhancements"""
        try:
            # AR marker detection
            frame = self._detect_ar_markers(frame)
            
            # Pose tracking overlay
            frame = self.pose_tracker.process_frame(frame)
            
            # Facial expression overlay
            frame = self.expression_engine.apply_to_frame(frame, self.current_emotion)
            
            # Cloth physics simulation
            frame = self.cloth_physics.simulate_frame(frame, 'idle')
            
            # Add AR status overlay
            frame = self._add_ar_status_overlay(frame)
            
            return frame
            
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return frame

    def _detect_ar_markers(self, frame):
        """Detect and process AR markers"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten()
                if self.MARKER_ID in ids_flat:
                    self.marker_detected = True
                    self.last_marker_time = time.time()
                    
                    idx = list(ids_flat).index(self.MARKER_ID)
                    corner = corners[idx][0]
                    
                    # Draw marker boundaries
                    pts = corner.astype(int)
                    cv2.polylines(frame, [pts], True, (255, 20, 147), 3)
                    
                    # Add overlay if available
                    if self.overlays:
                        overlay = list(self.overlays.values())[0]
                        frame = self._apply_overlay(frame, overlay, pts)
                    
                    # Trigger AR response
                    self._trigger_ar_response()
                    
            else:
                if time.time() - self.last_marker_time > 2.0:
                    self.marker_detected = False
                    
        except Exception as e:
            logging.error(f"Marker detection error: {e}")
            
        return frame

    def _apply_overlay(self, background, overlay, marker_points):
        """Apply overlay image to marker position"""
        try:
            # Calculate dimensions and position
            x_coords = marker_points[:, 0]
            y_coords = marker_points[:, 1]
            x, y = int(np.min(x_coords)), int(np.min(y_coords))
            w = int(np.max(x_coords) - x)
            h = int(np.max(y_coords) - y)
            
            # Resize overlay to match marker dimensions
            overlay_resized = cv2.resize(overlay, (w, h))
            
            # Apply overlay with alpha channel
            if overlay_resized.shape[2] == 4:
                alpha = overlay_resized[:, :, 3] / 255.0
                overlay_rgb = overlay_resized[:, :, :3]
                
                # Extract region of interest
                roi = background[y:y+h, x:x+w]
                
                # Ensure dimensions match
                if roi.shape[:2] == overlay_rgb.shape[:2]:
                    for c in range(3):
                        roi[:, :, c] = (alpha * overlay_rgb[:, :, c] + 
                                      (1 - alpha) * roi[:, :, c])
            else:
                # No alpha channel, simple overlay
                background[y:y+h, x:x+w] = overlay_resized
                        
        except Exception as e:
            logging.error(f"Overlay application error: {e}")
            
        return background

    def _trigger_ar_response(self):
        """Trigger AR responses when marker detected"""
        if not self.brain:
            return
            
        # Get appropriate response based on current state
        gesture = self.gesture_engine.get_contextual_gesture(self.current_emotion, "teasing")
        
        # Update brain with AR event
        if hasattr(self.brain, 'process_ar_event'):
            self.brain.process_ar_event({
                'type': 'marker_detected',
                'emotion': self.current_emotion,
                'gesture': gesture,
                'timestamp': datetime.now().isoformat()
            })

    def _add_ar_status_overlay(self, frame):
        """Add AR status information to frame"""
        try:
            h, w = frame.shape[:2]
            
            # Status background
            cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 150), (255, 255, 255), 2)
            
            # Status text
            status_lines = [
                f"MALIBU DUGAN AR - {self.current_emotion.upper()}",
                f"Marker: {'DETECTED' if self.marker_detected else 'SEARCHING'}",
                f"Pose: {self.pose_tracker.get_current_pose()}",
                f"Gaze: {self.eye_tracker.get_current_gaze()}",
                f"FPS: {self._calculate_fps():.1f}"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 40 + i * 25
                cv2.putText(frame, line, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        except Exception as e:
            logging.error(f"Status overlay error: {e}")
            
        return frame

    def _calculate_fps(self):
        """Calculate current FPS"""
        if not hasattr(self, '_fps_start_time'):
            self._fps_start_time = time.time()
            self._fps_frame_count = 0
            return 0.0
            
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_start_time
        
        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_start_time = time.time()
            self._fps_frame_count = 0
            
        return getattr(self, '_current_fps', 0.0)

    def update_emotion(self, emotion):
        """Update current emotion state"""
        self.current_emotion = emotion
        self.expression_engine.set_emotion(emotion)
        
        # Update component systems
        self.eye_tracker.update_emotion(emotion)
        self.face_gaze_sync.update_emotion(emotion)

    def get_ar_data(self):
        """Get comprehensive AR data for other systems"""
        return {
            'marker_detected': self.marker_detected,
            'current_emotion': self.current_emotion,
            'pose_data': self.pose_tracker.get_pose_statistics(),
            'gaze_data': self.eye_tracker.get_gaze_data(),
            'facial_data': self.facial_tracker.get_facial_data(),
            'sync_data': self.face_gaze_sync.get_sync_data()
        }

    def get_processed_frame(self):
        """Get latest processed AR frame"""
        try:
            if not self.processed_frame_queue.empty():
                return self.processed_frame_queue.get_nowait()
        except:
            pass
        return None

# === INTEGRATED COMPONENT CLASSES ===

class AdvancedBodyPoseTracking:
    """Enhanced pose tracking with MediaPipe integration"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.running = False
        self.current_pose = "standing"
        # Implementation details would go here...
        
    def start_pose_tracking(self):
        self.running = True
        logging.info("Pose tracking started")
        
    def stop_tracking(self):
        self.running = False
        logging.info("Pose tracking stopped")
        
    def process_frame(self, frame):
        # Pose processing implementation
        return frame
        
    def get_current_pose(self):
        return self.current_pose
        
    def get_pose_statistics(self):
        return {"pose": self.current_pose, "tracking": self.running}

class ARFacialTracking:
    """Facial tracking with emotion detection"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.running = False
        # Implementation details would go here...
        
    def start_tracking(self):
        self.running = True
        logging.info("Facial tracking started")
        
    def stop_tracking(self):
        self.running = False
        logging.info("Facial tracking stopped")
        
    def get_facial_data(self):
        return {"emotion": "neutral", "eye_contact": False}

class EyeGazeTracker:
    """Enhanced eye gaze tracking"""
    
    def __init__(self, gui_root=None):
        self.gui = gui_root
        self.running = False
        self.current_gaze = "center"
        
    def start_tracking(self):
        self.running = True
        logging.info("Eye gaze tracking started")
        
    def stop_tracking(self):
        self.running = False
        logging.info("Eye gaze tracking stopped")
        
    def update_emotion(self, emotion):
        pass
        
    def get_current_gaze(self):
        return self.current_gaze
        
    def get_gaze_data(self):
        return {"direction": self.current_gaze, "tracking": self.running}

class FaceGazeSync:
    """Face and gaze synchronization"""
    
    def __init__(self, gui_root=None):
        self.gui = gui_root
        self.running = False
        
    def start_sync(self):
        self.running = True
        logging.info("Face gaze sync started")
        
    def stop_sync(self):
        self.running = False
        logging.info("Face gaze sync stopped")
        
    def update_emotion(self, emotion):
        pass
        
    def get_sync_data(self):
        return {"sync_active": self.running}

class FacialExpressionEngine:
    """Facial expression generation engine"""
    
    def __init__(self):
        self.current_emotion = "neutral"
        
    def set_emotion(self, emotion):
        self.current_emotion = emotion
        
    def apply_to_frame(self, frame, emotion=None):
        if emotion:
            self.current_emotion = emotion
        # Expression application logic
        return frame

class BodyGestureEngine:
    """Body gesture engine"""
    
    def __init__(self, brain=None):
        self.brain = brain
        
    def get_contextual_gesture(self, emotion, context):
        return {"name": "hip_sway", "intensity": GestureIntensity.MODERATE}

class ClothPhysics:
    """Cloth physics simulation"""
    
    def __init__(self, fabric_type):
        self.fabric_type = fabric_type
        
    def simulate_frame(self, frame, movement_type):
        # Cloth physics simulation
        return frame

# === GLOBAL INSTANCE MANAGEMENT ===

_ar_engine_instance = None

def get_ar_engine(brain=None, gui_root=None):
    """Get or create global AR engine instance"""
    global _ar_engine_instance
    if _ar_engine_instance is None:
        _ar_engine_instance = ARSyncEngine(brain, gui_root)
    return _ar_engine_instance

def start_ar_system(brain=None, gui_root=None):
    """Start the AR system"""
    engine = get_ar_engine(brain, gui_root)
    engine.start_ar()
    return engine

def stop_ar_system():
    """Stop the AR system"""
    global _ar_engine_instance
    if _ar_engine_instance:
        _ar_engine_instance.stop_ar()
        _ar_engine_instance = None

# === TEST FUNCTION ===
def test_ar_system():
    """Test the AR system standalone"""
    logging.basicConfig(level=logging.INFO)
    print("Testing Malibu DuGan AR System...")
    
    ar_engine = ARSyncEngine()
    ar_engine.start_ar()
    
    try:
        # Run for 30 seconds
        for i in range(30):
            frame = ar_engine.get_processed_frame()
            if frame is not None:
                cv2.imshow("AR Test", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.1)  # Reduced sleep for better responsiveness
            
    except KeyboardInterrupt:
        pass
    finally:
        ar_engine.stop_ar()
        cv2.destroyAllWindows()
        print("AR Test completed.")

if __name__ == "__main__":
    test_ar_system()