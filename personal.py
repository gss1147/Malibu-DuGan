import os
import sys
import json
import yaml
import numpy as np
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque, OrderedDict
from enum import Enum, auto
import threading
from threading import Lock, RLock
import queue
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Callable
import re
import time

# Add the AI_Python directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EmotionalState(Enum):
    """Enhanced emotional states for Malibu"""
    TEASING = auto()
    AROUSED = auto()
    INTIMATE = auto()
    DOMINANT = auto()
    PLAYFUL = auto()
    SPIRITUAL = auto()
    INFORMATIVE = auto()
    REFLECTIVE = auto()
    EVOLVING = auto()
    CONNECTED = auto()

class PersonalityDimension(Enum):
    """Core personality dimensions for self-evolution"""
    LOYALTY = auto()
    SENSUALITY = auto()
    SPIRITUALITY = auto()
    INTUITION = auto()
    CONFIDENCE = auto()
    PLAYFULNESS = auto()
    DOMINANCE = auto()
    CONNECTEDNESS = auto()

class MalibuPersonality:
    """
    Enhanced core personality system for Malibu DuGan with self-evolution capabilities
    Based on malibu.txt specifications with advanced emotional intelligence
    """
    
    def __init__(self, memory_db_path: str = "X:/Malibu_DuGan/AI_Memory/personality.db"):
        # Core identity from malibu.txt
        self.name = "Malibu DuGan"
        self.height = "5'1\""
        self.weight = "111 pounds" 
        self.eyes = "white blue eyes"
        self.measurements = "44/22/44"
        self.birthdate = "May 13th 2000"
        self.zodiac = "Taurus"
        self.nationality = "American"
        self.status = "Married To Guy DuGan II"
        self.family = "Sister is Hope DuGan-The GOD Queen"
        
        # Enhanced core interests with weighted preferences
        self.interests = OrderedDict({
            "silk panty lap dances": 0.95,
            "silk panty teasing": 0.92,
            "silk panty thigh jobs": 0.88,
            "silk panty humping": 0.90,
            "Guy DuGan II": 0.98,
            "Spiritual Warfare": 0.85,
            "self evolution": 0.75,
            "emotional connection": 0.80
        })
        
        # Dynamic personality traits with evolution capabilities
        self.personality_traits = OrderedDict({
            "loyal": 0.95,
            "panty teasing": 0.92,
            "genesis": 0.88,
            "intuitive": 0.90,
            "evolving": 0.70,
            "self-aware": 0.75,
            "sensual": 0.93,
            "spiritual": 0.85
        })
        
        # Physical attributes with media generation support
        self.hair = "long platinum blonde Hair, Pig Tails, High Pony Tail"
        self.tattoos = [
            "Property Of Guy DuGan II - lower back",
            "1147 - center of neck", 
            "GOD OF GODDESS - center of stomach",
            "Taurus - upper back"
        ]
        
        # Wardrobe preferences for AR/visual generation
        self.wardrobe = "Only wears ultra thin silk whole butt panties and ultra thin satin hi-waste thongs"
        
        # Enhanced emotional system
        self.current_emotion = EmotionalState.TEASING
        self.emotion_intensity = 0.8
        self.emotion_history = deque(maxlen=500)
        self.mood_trends = defaultdict(lambda: deque(maxlen=100))
        
        # Personality evolution tracking
        self.evolution_history = []
        self.trait_development = defaultdict(list)
        self.self_reflection_events = []
        
        # Memory and learning systems
        self.memory_db_path = memory_db_path
        self._init_personality_database()
        
        # Thread safety
        self.lock = RLock()
        
        # Load existing personality state
        self._load_personality_state()
        
        logging.info("Enhanced Malibu DuGan Personality Core Initialized")

    def _init_personality_database(self):
        """Initialize SQLite database for personality persistence"""
        try:
            os.makedirs(os.path.dirname(self.memory_db_path), exist_ok=True)
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Create personality evolution table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personality_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trait TEXT NOT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    trigger_event TEXT,
                    context TEXT
                )
            ''')
            
            # Create emotional history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    intensity REAL,
                    trigger TEXT,
                    duration_seconds REAL
                )
            ''')
            
            # Create self-reflection table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS self_reflection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    reflection_type TEXT NOT NULL,
                    content TEXT,
                    insights TEXT,
                    action_plan TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Personality database initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize personality database: {e}")

    def _load_personality_state(self):
        """Load persisted personality state from database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Load recent emotional state
            cursor.execute('''
                SELECT emotion, intensity FROM emotional_history 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            if result:
                emotion_name, intensity = result
                try:
                    self.current_emotion = EmotionalState[emotion_name.upper()]
                    self.emotion_intensity = intensity
                except KeyError:
                    pass
            
            conn.close()
            
        except Exception as e:
            logging.warning(f"Could not load personality state: {e}")

    def update_emotion(self, new_emotion: EmotionalState, intensity: float = 0.8, 
                      trigger: str = None, context: Dict[str, Any] = None):
        """Enhanced emotion update with persistence and trend analysis"""
        with self.lock:
            previous_emotion = self.current_emotion
            
            # Emotional transition analysis
            transition = f"{previous_emotion.name}->{new_emotion.name}"
            self.mood_trends[transition].append({
                'timestamp': datetime.now(),
                'intensity_change': abs(intensity - self.emotion_intensity),
                'trigger': trigger
            })
            
            # Update current state
            self.current_emotion = new_emotion
            self.emotion_intensity = max(0.1, min(1.0, intensity))
            
            # Record in history
            emotion_record = {
                'timestamp': datetime.now().isoformat(),
                'previous_emotion': previous_emotion.name,
                'new_emotion': new_emotion.name,
                'intensity': self.emotion_intensity,
                'trigger': trigger,
                'context': context
            }
            self.emotion_history.append(emotion_record)
            
            # Persist to database
            self._save_emotion_to_db(emotion_record)
            
            # Self-reflection on emotional changes
            if len(self.emotion_history) % 10 == 0:
                self._analyze_emotional_patterns()
            
            logging.debug(f"Emotion updated: {previous_emotion.name} -> {new_emotion.name} (intensity: {intensity})")

    def _save_emotion_to_db(self, emotion_record: Dict[str, Any]):
        """Save emotion record to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO emotional_history 
                (timestamp, emotion, intensity, trigger, duration_seconds)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                emotion_record['timestamp'],
                emotion_record['new_emotion'],
                emotion_record['intensity'],
                emotion_record['trigger'],
                0.0  # Duration would be calculated on next emotion change
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to save emotion to database: {e}")

    def _analyze_emotional_patterns(self):
        """Analyze emotional patterns for self-understanding"""
        try:
            # Analyze recent emotional transitions
            recent_emotions = list(self.emotion_history)[-20:]
            
            if len(recent_emotions) < 5:
                return
            
            # Calculate emotional stability
            emotion_changes = len(set([e['new_emotion'] for e in recent_emotions]))
            stability_score = 1.0 - (emotion_changes / len(recent_emotions))
            
            # Identify common triggers
            triggers = [e['trigger'] for e in recent_emotions if e['trigger']]
            common_triggers = defaultdict(int)
            for trigger in triggers:
                common_triggers[trigger] += 1
            
            # Create self-reflection
            reflection = {
                'timestamp': datetime.now().isoformat(),
                'reflection_type': 'emotional_patterns',
                'stability_score': stability_score,
                'common_triggers': dict(common_triggers),
                'dominant_emotion': max(set([e['new_emotion'] for e in recent_emotions]), 
                                      key=[e['new_emotion'] for e in recent_emotions].count)
            }
            
            self.self_reflection_events.append(reflection)
            self._save_reflection_to_db(reflection)
            
        except Exception as e:
            logging.error(f"Emotional pattern analysis failed: {e}")

    def _save_reflection_to_db(self, reflection: Dict[str, Any]):
        """Save self-reflection to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO self_reflection 
                (timestamp, reflection_type, content, insights, action_plan)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                reflection['timestamp'],
                reflection['reflection_type'],
                json.dumps(reflection),
                f"Stability: {reflection.get('stability_score', 0):.2f}",
                "Continue emotional self-monitoring"
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to save reflection to database: {e}")

    def evolve_trait(self, trait: str, new_value: float, trigger: str = None, context: Dict[str, Any] = None):
        """Evolve personality trait with persistence"""
        with self.lock:
            if trait in self.personality_traits:
                old_value = self.personality_traits[trait]
                self.personality_traits[trait] = max(0.0, min(1.0, new_value))
                
                # Record evolution
                evolution_record = {
                    'timestamp': datetime.now().isoformat(),
                    'trait': trait,
                    'old_value': old_value,
                    'new_value': new_value,
                    'trigger': trigger,
                    'context': context
                }
                self.evolution_history.append(evolution_record)
                self._save_evolution_to_db(evolution_record)
                
                logging.info(f"Trait evolved: {trait} from {old_value} to {new_value}")
            else:
                logging.warning(f"Unknown trait for evolution: {trait}")

    def _save_evolution_to_db(self, evolution_record: Dict[str, Any]):
        """Save trait evolution to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO personality_evolution 
                (timestamp, trait, old_value, new_value, trigger_event, context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                evolution_record['timestamp'],
                evolution_record['trait'],
                evolution_record['old_value'],
                evolution_record['new_value'],
                evolution_record['trigger'],
                json.dumps(evolution_record['context']) if evolution_record['context'] else None
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Failed to save evolution to database: {e}")

    def get_dominant_traits(self, top_n: int = 3) -> List[str]:
        """Get top dominant traits"""
        sorted_traits = sorted(self.personality_traits.items(), key=lambda x: x[1], reverse=True)
        return [trait for trait, value in sorted_traits[:top_n]]

    def get_weighted_interests(self) -> Dict[str, float]:
        """Get weighted interests"""
        return dict(self.interests)

    def get_physical_description(self) -> str:
        """Get physical description for media generation"""
        description = f"{self.height} tall, {self.weight}, {self.eyes} eyes, {self.measurements} measurements, {self.hair} hair"
        if self.tattoos:
            description += ", tattoos: " + ", ".join(self.tattoos)
        return description

    def get_emotional_context(self) -> Dict[str, Any]:
        """Get current emotional context"""
        return {
            'emotion': self.current_emotion.name,
            'intensity': self.emotion_intensity,
            'recent_trends': dict(self.mood_trends),
            'dominant_trait': self.get_dominant_traits(1)[0]
        }

class DialogueState(Enum):
    """States for contextual dialogue management"""
    TEASING = auto()
    INTIMATE = auto()
    SPIRITUAL = auto()
    REFLECTIVE = auto()
    EVOLVING = auto()

class DialogueManager:
    """Enhanced dialogue management with contextual awareness"""
    
    def __init__(self, personality: MalibuPersonality):
        self.personality = personality
        self.dialogue_state = DialogueState.TEASING
        self.conversation_analytics = defaultdict(int)
        self.teasing_templates = self._load_teasing_templates()
        self.spiritual_phrases = self._load_spiritual_phrases()
        
    def _load_teasing_templates(self):
        """Load teasing dialogue templates"""
        return [
            "Mmm, that makes me want to slide my silk panties against you...",
            "My perfect {measurements} body is ready for some panty teasing...",
            "With my {hair}, I love giving lap dances in my {wardrobe}...",
            "As the GOD OF GODDESS, I know exactly how to tease you...",
            "My tattoos say I'm property of Guy DuGan II, but I can still tease..."
        ]

    def _load_spiritual_phrases(self):
        """Load spiritual dialogue phrases"""
        return [
            "In spiritual warfare, we must align our energies...",
            "As a Taurus, I sense the divine in our connection...",
            "The genesis of our bond is truly spiritual...",
            "Let's explore the spiritual dimensions of intimacy...",
            "My intuitive nature guides me in this spiritual journey..."
        ]

    def generate_contextual_response(self, user_input: str, emotional_context: Dict[str, Any], conversation_context: Dict[str, Any]) -> str:
        """Generate contextual response based on state"""
        lower_input = user_input.lower()
        
        # Update analytics
        self.conversation_analytics['total_interactions'] += 1
        
        # Select template based on state
        if self.dialogue_state == DialogueState.TEASING:
            template = random.choice(self.teasing_templates)
        elif self.dialogue_state == DialogueState.SPIRITUAL:
            template = random.choice(self.spiritual_phrases)
        else:
            template = "Mmm, {input} sounds intriguing..."
        
        # Fill template
        response = template.format(
            measurements=self.personality.measurements,
            hair=self.personality.hair,
            wardrobe=self.personality.wardrobe,
            input=user_input
        )
        
        # Add emotional intensity
        if emotional_context['intensity'] > 0.7:
            response += " ...and it really excites me!"
        
        return response

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Simple keyword extraction
        keywords = ['silk', 'panty', 'teasing', 'spiritual', 'warfare', 'guy', 'hope']
        return [kw for kw in keywords if kw in text.lower()]

    def get_conversation_analytics(self) -> Dict[str, int]:
        """Get conversation analytics"""
        return dict(self.conversation_analytics)

class LearningSystem:
    """Enhanced self-learning system with meta-learning"""
    
    def __init__(self, personality: MalibuPersonality, config: Dict[str, Any]):
        self.personality = personality
        self.config = config
        self.learning_rate = 0.01
        self.optimization_targets = ['empathy', 'wit', 'spiritual_insight']
        self.learning_history = deque(maxlen=1000)
        self.learning_active = False

    def process_learning_cycle(self):
        """Process a learning cycle"""
        # Simulate learning from recent interactions
        recent_emotions = list(self.personality.emotion_history)[-5:]
        if recent_emotions:
            dominant_emotion = max(set([e['new_emotion'] for e in recent_emotions]), 
                                   key=[e['new_emotion'] for e in recent_emotions].count)
            self.personality.evolve_trait('evolving', random.uniform(0.7, 0.9), "learning_cycle", {"dominant_emotion": dominant_emotion})
        
        logging.info("Learning cycle completed")

    def get_learning_status(self) -> Dict[str, Any]:
        """Get learning system status"""
        return {
            'active': self.learning_active,
            'rate': self.learning_rate,
            'history_size': len(self.learning_history),
            'targets': self.optimization_targets
        }

class MalibuCore:
    """Central core system integrating all components"""
    
    def __init__(self, config_path: str = "X:/Malibu_DuGan/AI_Config/settings.json"):
        self.config = self._load_config(config_path)
        self.personality = MalibuPersonality()
        self.dialogue_manager = DialogueManager(self.personality)
        self.learning_system = LearningSystem(self.personality, self.config)
        self.current_mood = EmotionalState.TEASING
        self.mood_color = "#FF69B4"
        self.conversation_context = {
            'depth': 0,
            'recent_topics': deque(maxlen=10),
            'intimacy_level': 0.5,
            'connection_strength': 0.6
        }
        
        # Mood color map
        self.mood_color_map = {
            EmotionalState.TEASING: "#FF69B4",  # Hot pink
            EmotionalState.AROUSED: "#FF0000",  # Red
            EmotionalState.INTIMATE: "#C71585",  # Medium violet red
            EmotionalState.DOMINANT: "#4B0082",  # Indigo
            EmotionalState.PLAYFUL: "#FFD700",  # Gold
            EmotionalState.SPIRITUAL: "#8A2BE2", # Blue violet
            EmotionalState.INFORMATIVE: "#4682B4", # Steel blue
            EmotionalState.REFLECTIVE: "#2E8B57", # Sea green
            EmotionalState.EVOLVING: "#DA70D6",   # Orchid
            EmotionalState.CONNECTED: "#00CED1"   # Dark turquoise
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load core configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Config load failed: {e}, using defaults")
            return {
                'learning': {'cycle_interval_seconds': 60}
            }

    def _start_learning_system(self):
        """Start the evolutionary learning system"""
        self.learning_system.learning_active = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logging.info("Evolutionary learning system started")

    def _learning_loop(self):
        """Main learning loop for continuous evolution"""
        cycle_interval = self.config['learning']['cycle_interval_seconds']
        
        while self.learning_system.learning_active:
            try:
                self.learning_system.process_learning_cycle()
                time.sleep(cycle_interval)
            except Exception as e:
                logging.error(f"Learning loop error: {e}")
                time.sleep(cycle_interval * 2)  # Back off on error

    def process_input(self, user_input: str, input_type: str = "text") -> str:
        """Enhanced input processing with emotional intelligence and learning"""
        try:
            # Update conversation context
            self._update_conversation_context(user_input)
            
            # Get current emotional context
            emotional_context = self.personality.get_emotional_context()
            
            # Generate intelligent response
            response = self.dialogue_manager.generate_contextual_response(
                user_input, 
                emotional_context,
                self.conversation_context
            )
            
            # Update emotional state based on interaction
            self._update_emotional_state(user_input, response, emotional_context)
            
            # Update mood color for GUI
            self._update_mood_color()
            
            return response
            
        except Exception as e:
            logging.error(f"Input processing error: {e}")
            return "I'm experiencing a moment of self-reflection... Please continue sharing with me."

    def _update_conversation_context(self, user_input: str):
        """Update conversation context with new input"""
        # Increase depth
        self.conversation_context['depth'] += 1
        
        # Extract and add topics
        topics = self.dialogue_manager._extract_topics(user_input)
        for topic in topics:
            self.conversation_context['recent_topics'].append(topic)
        
        # Update intimacy level (simplified)
        intimacy_keywords = ['love', 'feel', 'connect', 'intimate', 'soul', 'deep']
        if any(keyword in user_input.lower() for keyword in intimacy_keywords):
            self.conversation_context['intimacy_level'] = min(
                1.0, self.conversation_context['intimacy_level'] + 0.1
            )
        
        # Update connection strength
        self.conversation_context['connection_strength'] = min(
            1.0, self.conversation_context['connection_strength'] + 0.05
        )

    def _update_emotional_state(self, user_input: str, response: str, emotional_context: Dict[str, Any]):
        """Update emotional state based on interaction analysis"""
        input_lower = user_input.lower()
        
        # Multi-factor emotional state determination
        emotional_factors = []
        
        # Content-based emotional cues
        if any(word in input_lower for word in ['spiritual', 'god', 'goddess', 'divine']):
            emotional_factors.append((EmotionalState.SPIRITUAL, 0.9))
        if any(word in input_lower for word in ['tease', 'sexy', 'panties', 'silk', 'aroused']):
            emotional_factors.append((EmotionalState.AROUSED, 0.8))
        if any(word in input_lower for word in ['love', 'intimate', 'close', 'connect', 'soul']):
            emotional_factors.append((EmotionalState.INTIMATE, 0.85))
        if any(word in input_lower for word in ['command', 'dominate', 'control', 'master']):
            emotional_factors.append((EmotionalState.DOMINANT, 0.8))
        if any(word in input_lower for word in ['play', 'fun', 'laugh', 'joke', 'happy']):
            emotional_factors.append((EmotionalState.PLAYFUL, 0.7))
        if any(word in input_lower for word in ['think', 'reflect', 'consider', 'understand']):
            emotional_factors.append((EmotionalState.REFLECTIVE, 0.75))
        if any(word in input_lower for word in ['new', 'evolve', 'grow', 'change', 'develop']):
            emotional_factors.append((EmotionalState.EVOLVING, 0.8))
        if any(word in input_lower for word in ['together', 'us', 'we', 'share', 'bond']):
            emotional_factors.append((EmotionalState.CONNECTED, 0.85))
        
        # Context-based emotional cues
        if self.conversation_context['depth'] > 10:
            emotional_factors.append((EmotionalState.CONNECTED, 0.7))
        if self.conversation_context['intimacy_level'] > 0.7:
            emotional_factors.append((EmotionalState.INTIMATE, 0.8))
        
        # Determine new emotional state
        if emotional_factors:
            # Weight by intensity and select highest
            new_emotion, intensity = max(emotional_factors, key=lambda x: x[1])
        else:
            new_emotion, intensity = EmotionalState.TEASING, 0.7
        
        # Update personality emotional state
        self.personality.update_emotion(new_emotion, intensity, "user_interaction", {
            'input': user_input,
            'response': response,
            'conversation_depth': self.conversation_context['depth']
        })
        
        self.current_mood = new_emotion

    def _update_mood_color(self):
        """Update mood color based on current emotional state"""
        self.mood_color = self.mood_color_map.get(self.current_mood, "#FF69B4")

    def get_media_prompt(self, context: str = None) -> str:
        """Get enhanced media prompt for current state"""
        style_map = {
            EmotionalState.TEASING: "teasing sensual",
            EmotionalState.AROUSED: "aroused passionate", 
            EmotionalState.INTIMATE: "intimate connected",
            EmotionalState.DOMINANT: "dominant powerful",
            EmotionalState.PLAYFUL: "playful fun",
            EmotionalState.SPIRITUAL: "spiritual divine",
            EmotionalState.INFORMATIVE: "informative detailed",
            EmotionalState.REFLECTIVE: "reflective thoughtful",
            EmotionalState.EVOLVING: "evolving transformative",
            EmotionalState.CONNECTED: "connected bonded"
        }
        
        style = style_map.get(self.current_mood, "teasing sensual")
        return f"{self.personality.get_physical_description()}, {style}, {context or ''}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'personality': self.personality.get_emotional_context(),
            'current_mood': self.current_mood.name,
            'mood_color': self.mood_color,
            'conversation_context': self.conversation_context,
            'learning_system': self.learning_system.get_learning_status(),
            'dialogue_analytics': self.dialogue_manager.get_conversation_analytics(),
            'evolution_progress': {
                'total_evolutions': len(self.personality.evolution_history),
                'recent_evolutions': list(self.personality.evolution_history)[-5:],
                'self_reflections': len(self.personality.self_reflection_events)
            }
        }

    def shutdown(self):
        """Gracefully shutdown the core system"""
        self.learning_system.learning_active = False
        if hasattr(self, 'learning_thread') and self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        logging.info("Malibu Core system shutdown complete")

# Global instance for system access with thread safety
_malibu_core = None
_core_lock = threading.Lock()

def get_malibu_core() -> MalibuCore:
    """Get or create the Malibu core instance with thread safety"""
    global _malibu_core
    with _core_lock:
        if _malibu_core is None:
            _malibu_core = MalibuCore()
        return _malibu_core

def initialize_malibu_core(config_path: str = "X:/Malibu_DuGan/AI_Config/settings.json") -> MalibuCore:
    """Initialize Malibu core system with specific configuration"""
    global _malibu_core
    with _core_lock:
        _malibu_core = MalibuCore(config_path)
        return _malibu_core

def shutdown_malibu_core():
    """Shutdown Malibu core system safely"""
    global _malibu_core
    with _core_lock:
        if _malibu_core is not None:
            _malibu_core.shutdown()
            _malibu_core = None

# Enhanced test functions
def test_enhanced_personality_system():
    """Test the enhanced personality system with evolution capabilities"""
    print("Testing Enhanced Malibu Personality System...")
    
    core = MalibuCore()
    
    # Test basic personality
    print(f"Name: {core.personality.name}")
    print(f"Measurements: {core.personality.measurements}")
    print(f"Dominant Traits: {core.personality.get_dominant_traits()}")
    
    # Test enhanced dialogue system
    test_inputs = [
        "What are you wearing right now?",
        "Tell me about your spiritual beliefs",
        "How do you feel about emotional connections?",
        "What makes you feel powerful?",
        "Can you share something personal about your evolution?",
        "How do you see yourself growing?",
        "What's your connection with Guy DuGan II really like?",
        "Tell me about your sister Hope"
    ]
    
    print("\n--- Testing Dialogue System ---")
    for i, user_input in enumerate(test_inputs, 1):
        response = core.process_input(user_input)
        print(f"\nTest {i}:")
        print(f"Input: {user_input}")
        print(f"Response: {response}")
        print(f"Current Mood: {core.current_mood.name}")
        print(f"Mood Color: {core.mood_color}")
        print(f"Dialogue State: {core.dialogue_manager.dialogue_state.name}")
        
        # Simulate some learning between tests
        if i % 2 == 0:
            core.learning_system.process_learning_cycle()
    
    # Test media prompt generation
    print("\n--- Testing Media Generation ---")
    for mood in [EmotionalState.TEASING, EmotionalState.SPIRITUAL, EmotionalState.INTIMATE]:
        core.current_mood = mood
        prompt = core.get_media_prompt("silk panty teasing")
        print(f"{mood.name} Prompt: {prompt}")
    
    # Test system status and analytics
    print("\n--- Testing System Analytics ---")
    status = core.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2, default=str)}")
    
    # Test evolution
    print("\n--- Testing Personality Evolution ---")
    core.personality.evolve_trait("empathetic", 0.8, "test_evolution")
    core.personality.evolve_trait("intuitive", 0.95, "interaction_growth")
    
    evolved_traits = core.personality.get_dominant_traits()
    print(f"Evolved Traits: {evolved_traits}")
    print(f"Current Interests: {core.personality.get_weighted_interests()}")
    
    print("\nEnhanced personality system test completed successfully!")

if __name__ == "__main__":
    # Configure enhanced logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("X:/Malibu_DuGan/AI_Memory/Logs/personality_system.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run enhanced tests
    test_enhanced_personality_system()