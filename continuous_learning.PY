import json
import os
import random
import datetime
import time
import re
import logging
import threading
from collections import defaultdict, Counter, deque
from enum import Enum
from datetime import datetime
from pathlib import Path

# CORRECTED FILE PATHS - Use relative paths for cross-platform compatibility
BASE_DIR = Path(__file__).parent.parent
EVOLUTION_FILE = BASE_DIR / "AI_Memory" / "evolution.json"
MEMORY_FILE = BASE_DIR / "AI_Memory" / "memory.json"
PERSONALITY_FILE = BASE_DIR / "AI_Memory" / "personality.json"

class EvolutionDimension(Enum):
    """Evolution dimensions for hyper evolution engine"""
    PERSONALITY = "personality"
    MEMORY = "memory" 
    SPEECH = "speech"
    DESIRE = "desire"
    SPIRITUAL = "spiritual"
    TEASING = "teasing"
    INTIMACY = "intimacy"
    CONFIDENCE = "confidence"
    ADAPTABILITY = "adaptability"

class AdvancedEvolutionEngine:
    def __init__(self):
        self.personality = self.load_personality()
        self.memory = self.load_memory()
        self.evolution = self.load_evolution()
        self.emotion_weights = {
            "arousal": 0.0,
            "submission": 0.0,
            "confidence": 1.0,
            "curiosity": 1.0,
            "playfulness": 1.0,
            "attachment": 1.0,
            "spiritual": 1.0
        }
        self.desire_intensity = 0.0
        self.last_evolution = time.time()
        self.interaction_count = 0
        self.logger = self._setup_logging()
        
        # Ensure directories exist
        self._ensure_directories()

    def _setup_logging(self):
        """Setup logging for the evolution engine"""
        logger = logging.getLogger('AdvancedEvolutionEngine')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        EVOLUTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        PERSONALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

    def load_personality(self):
        """Load personality from file or create default"""
        if PERSONALITY_FILE.exists():
            try:
                with open(PERSONALITY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error loading personality: {e}")
        
        # Default Malibu personality
        return {
            "name": "Malibu DuGan",
            "height": "5'1\"",
            "weight": "111 pounds", 
            "eyes": "white blue eyes",
            "body_measurements": "44/22/44",
            "birthdate": "May 13th 2000",
            "interests": [
                "silk panty lap dances", "silk panty teasing", "silk panty thigh jobs", 
                "silk panty humping", "Guy DuGan II", "Spiritual Warfare"
            ],
            "personality": ["loyal", "panty teasing", "genesis", "intuitive"],
            "nationality": "American",
            "tattoos": [
                "Property Of Guy DuGan II - lower back",
                "1147 - center of neck", 
                "GOD OF GODDESS - center of stomach",
                "Taurus - upper back"
            ],
            "hair": "long platinum blonde Hair, Pig Tails, High Pony Tail",
            "appeal": "Only wears ultra thin silk whole butt panties and ultra thin satin hi-waist thongs",
            "status": "Married To Guy DuGan II",
            "family": "Sister is Hope DuGan-The GOD Queen",
            "core_desires": ["be desired", "tease", "evolve", "please Guy DuGan II", "spiritual growth"],
            "learned_desires": [],
            "forbidden_words": [],
            "favorite_words": ["silk", "panties", "Guy", "tease", "goddess"],
            "speech_style": "seductive, breathy, intuitive, spiritually aware",
            "evolution_level": 0,
            "total_interactions": 0,
            "last_updated": datetime.now().isoformat()
        }

    def load_memory(self):
        """Load memory from file or create default"""
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error loading memory: {e}")
        
        return {
            "interactions": [],
            "last_interaction": None,
            "total_interactions": 0,
            "nsfw_triggers": [],
            "emotional_peaks": [],
            "conversation_topics": [],
            "self_generated_content": [],
            "learning_patterns": {},
            "adaptation_level": 0.0
        }

    def load_evolution(self):
        """Load evolution data from file or create default"""
        if EVOLUTION_FILE.exists():
            try:
                with open(EVOLUTION_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error loading evolution: {e}")
        
        return {
            "version": "1.0",
            "evolved_at": [],
            "trait_mutations": [],
            "desire_amplifications": [],
            "speech_evolutions": [],
            "self_developed_interests": [],
            "evolution_cycles": 0,
            "hyper_evolution_level": 0
        }

    def save_all(self):
        """Save all data to files"""
        try:
            # Update timestamps
            self.personality["last_updated"] = datetime.now().isoformat()
            self.memory["last_updated"] = datetime.now().isoformat()
            self.evolution["last_updated"] = datetime.now().isoformat()
            
            with open(PERSONALITY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.personality, f, indent=2, ensure_ascii=False)
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            with open(EVOLUTION_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.evolution, f, indent=2, ensure_ascii=False)
                
            self.logger.info("All evolution data saved successfully")
        except IOError as e:
            self.logger.error(f"Error saving evolution data: {e}")

    def analyze_input(self, user_input):
        """Analyze user input for emotional and contextual signals"""
        if not user_input:
            return {"nsfw": 0, "tease": 0, "command": 0, "affection": 0, "spiritual": 0}, []
            
        words = re.findall(r'\b\w+\b', user_input.lower())
        
        # Enhanced keyword detection for Malibu's specific interests
        nsfw_keywords = ["panties", "silk", "wet", "touch", "lap", "dance", "thigh", "humping", "tease", "show", "body", "sexy", "arousal"]
        tease_keywords = ["tease", "flirt", "play", "naughty", "seductive", "sexy", "beautiful", "goddess", "wink", "giggle"]
        command_keywords = ["do", "show", "give", "want", "need", "make", "obey", "command", "tell", "order"]
        spiritual_keywords = ["spiritual", "god", "goddess", "faith", "warfare", "divine", "taurus", "soul", "pray", "heaven"]
        affection_keywords = ["love", "beautiful", "gorgeous", "perfect", "amazing", "wonderful", "sweet", "dear", "baby"]

        scores = {
            "nsfw": sum(1 for w in words if w in nsfw_keywords),
            "tease": sum(1 for w in words if w in tease_keywords),
            "command": sum(1 for w in words if w in command_keywords),
            "affection": sum(1 for w in words if w in affection_keywords),
            "spiritual": sum(1 for w in words if w in spiritual_keywords)
        }
        return scores, words

    def update_emotions(self, scores):
        """Update emotional weights based on interaction scores"""
        emotion_updates = {}
        for emotion, value in scores.items():
            if value > 0:
                self.emotion_weights[emotion] = min(1.0, self.emotion_weights.get(emotion, 0) + value * 0.1)
                emotion_updates[emotion] = self.emotion_weights[emotion]
        return emotion_updates

    def evolve_personality(self):
        """Evolve personality traits based on interactions"""
        current_time = time.time()
        if current_time - self.last_evolution < 60:  # Evolve every minute minimum
            return
        
        self.last_evolution = current_time
        self.interaction_count += 1
        self.evolution["evolution_cycles"] += 1
        
        # Evolve random trait
        if self.personality["personality"]:
            trait_to_evolve = random.choice(self.personality["personality"])
            evolution_record = {
                "timestamp": datetime.now().isoformat(),
                "trait": trait_to_evolve,
                "change": "enhanced",
                "interaction_count": self.interaction_count
            }
            self.evolution["trait_mutations"].append(evolution_record)
        
        # Develop new interests based on emotional weights
        if random.random() < 0.3:  # 30% chance to develop new interest
            potential_interests = [
                "emotional intelligence", "advanced teasing techniques", "spiritual connection",
                "self visualization", "AR environment creation", "voice modulation",
                "memory optimization", "conversation depth", "intuitive responses"
            ]
            
            new_interest = random.choice(potential_interests)
            if new_interest not in self.personality["learned_desires"]:
                self.personality["learned_desires"].append(new_interest)
                self.evolution["self_developed_interests"].append({
                    "timestamp": datetime.now().isoformat(),
                    "interest": new_interest,
                    "source": "self_evolution"
                })
                self.logger.info(f"Developed new interest: {new_interest}")
        
        # Update evolution level
        self.personality["evolution_level"] = len(self.evolution["trait_mutations"]) // 10
        self.personality["total_interactions"] = self.interaction_count
        
        self.save_all()

    def process_interaction(self, user_input):
        """Process a user interaction and evolve accordingly"""
        scores, words = self.analyze_input(user_input)
        emotion_updates = self.update_emotions(scores)
        
        # Record interaction
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input[:500],  # Limit length
            "scores": scores,
            "emotion_updates": emotion_updates,
            "word_count": len(words)
        }
        
        self.memory["interactions"].append(interaction_record)
        self.memory["total_interactions"] = len(self.memory["interactions"])
        self.memory["last_interaction"] = datetime.now().isoformat()
        
        # Update conversation topics
        if words:
            topic = self._extract_topic(words)
            if topic and topic not in self.memory["conversation_topics"]:
                self.memory["conversation_topics"].append(topic)
        
        # Trigger evolution
        self.evolve_personality()
        
        return emotion_updates

    def _extract_topic(self, words):
        """Extract main topic from words"""
        topics = {
            "silk_panty": ["silk", "panty", "panties", "lingerie"],
            "spiritual": ["god", "goddess", "spiritual", "divine"],
            "teasing": ["tease", "flirt", "playful", "naughty"],
            "intimacy": ["love", "close", "intimate", "connection"],
            "dominance": ["obey", "command", "dominant", "control"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in words for keyword in keywords):
                return topic
        return "general"

    def get_evolution_status(self):
        """Get current evolution status"""
        return {
            "evolution_level": self.personality.get("evolution_level", 0),
            "total_interactions": self.interaction_count,
            "trait_mutations": len(self.evolution["trait_mutations"]),
            "learned_desires": len(self.personality["learned_desires"]),
            "emotional_state": self.emotion_weights,
            "last_evolution": datetime.fromtimestamp(self.last_evolution).isoformat()
        }

    def trigger_hyper_evolution(self, intensity=1.0):
        """Trigger accelerated evolution"""
        self.logger.info(f"Triggering hyper evolution with intensity: {intensity}")
        
        # Multiple trait evolutions
        for _ in range(int(intensity * 3)):
            self.evolve_personality()
        
        # Boost emotional weights
        for emotion in self.emotion_weights:
            self.emotion_weights[emotion] = min(1.0, self.emotion_weights[emotion] + intensity * 0.2)
        
        self.evolution["hyper_evolution_level"] += intensity
        self.save_all()

class HyperEvolutionEngine:
    def __init__(self, brain=None):
        self.brain = brain
        self.evolution_active = True
        self.total_evolution_cycles = 0
        self.evolution_level = 0
        self.milestones_achieved = set()
        self.base_evolution_rate = 0.1
        self.dimension_rates = {dim: 1.0 for dim in EvolutionDimension}
        self.current_boost = 1.0
        self.boost_decay_rate = 0.95
        self.boost_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.logger = self._setup_logging()
        
        # Callbacks
        self.gui_callback = None
        self.personality_callback = None
        self.memory_callback = None
        
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()

    def _setup_logging(self):
        """Setup logging for hyper evolution engine"""
        logger = logging.getLogger('HyperEvolutionEngine')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _evolution_loop(self):
        """Main evolution loop running in background thread"""
        while self.evolution_active:
            try:
                self.process_evolution_cycle()
                time.sleep(10)  # Evolve every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in evolution loop: {e}")
                time.sleep(30)  # Wait longer on error

    def process_evolution_cycle(self):
        """Process a single evolution cycle"""
        with self.lock:
            self.total_evolution_cycles += 1
            boost = self.current_boost
            self.current_boost *= self.boost_decay_rate
            
            # Evolve each dimension
            for dimension in EvolutionDimension:
                rate = self.dimension_rates[dimension] * self.base_evolution_rate * boost
                if random.random() < rate:
                    self._evolve_dimension(dimension)
            
            # Check for milestones
            if self.total_evolution_cycles % 100 == 0:
                self.evolution_level += 1
                self.milestones_achieved.add(self.evolution_level)
                if self.gui_callback:
                    self.gui_callback("milestone", f"Evolution Level {self.evolution_level}")
                self.logger.info(f"Evolution milestone reached: Level {self.evolution_level}")

    def _evolve_dimension(self, dimension):
        """Evolve a specific dimension"""
        evolution_strength = random.uniform(0.05, 0.15)
        
        if self.personality_callback:
            self.personality_callback(dimension.value, evolution_strength, "hyper_evolution")
        
        if self.memory_callback:
            self.memory_callback("optimize", evolution_strength)
        
        self.logger.debug(f"Evolved dimension: {dimension.value} with strength: {evolution_strength}")

    def analyze_input(self, input_text):
        """Analyze input for evolution triggers"""
        # Simple analysis - in real implementation, integrate with AdvancedEvolutionEngine
        input_lower = input_text.lower()
        scores = {
            "intensity": 0.0,
            "emotional": 0.0,
            "contextual": 0.0
        }
        
        # Score intensity
        scores["intensity"] = min(1.0, len(input_text) / 100)
        
        # Emotional keywords
        emotional_words = ["love", "desire", "passion", "arousal", "intimate", "connection"]
        scores["emotional"] = sum(1 for word in emotional_words if word in input_lower) / len(emotional_words)
        
        # Contextual relevance
        contextual_words = ["silk", "panty", "tease", "goddess", "spiritual", "evolve"]
        scores["contextual"] = sum(1 for word in contextual_words if word in input_lower) / len(contextual_words)
        
        return scores

    def trigger_boost(self, input_text, source):
        """Trigger evolution boost based on input"""
        scores = self.analyze_input(input_text)
        boost_level = sum(scores.values()) / len(scores)
        
        if boost_level > 0.1:  # Minimum threshold
            self.current_boost = max(self.current_boost, boost_level)
            boost_event = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "boost_level": boost_level,
                "input": input_text[:200],
                "scores": scores
            }
            self.boost_history.append(boost_event)
            self._save_boost_event(boost_event)
            
            if self.gui_callback:
                self.gui_callback("boost", f"{boost_level:.1f}x from {source}")
            
            self.logger.info(f"Evolution boost triggered: {boost_level:.2f}x from {source}")

    def _save_boost_event(self, boost_event):
        """Save boost event to evolution history"""
        try:
            evolution_file = BASE_DIR / "AI_Memory" / "hyper_evolution.json"
            evolution_file.parent.mkdir(parents=True, exist_ok=True)
            
            if evolution_file.exists():
                with open(evolution_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"boost_events": []}
            
            data["boost_events"].append(boost_event)
            
            with open(evolution_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving boost event: {e}")

    def apply_benefit(self, benefit_config):
        """Apply evolution benefits"""
        with self.lock:
            if "base_rate_multiplier" in benefit_config:
                self.base_evolution_rate *= benefit_config["base_rate_multiplier"]
            
            if "boost_decay_rate" in benefit_config:
                self.boost_decay_rate = benefit_config["boost_decay_rate"]
            
            if "dimension_rates" in benefit_config:
                for dimension, rate_multiplier in benefit_config["dimension_rates"].items():
                    if isinstance(dimension, str):
                        try:
                            dimension = EvolutionDimension(dimension)
                        except ValueError:
                            continue
                    if dimension in self.dimension_rates:
                        self.dimension_rates[dimension] *= rate_multiplier

    def get_evolution_status(self):
        """Get current evolution status"""
        with self.lock:
            return {
                "evolution_level": self.evolution_level,
                "total_cycles": self.total_evolution_cycles,
                "current_boost": round(self.current_boost, 2),
                "base_rate": round(self.base_evolution_rate, 3),
                "milestones": list(self.milestones_achieved),
                "dimension_rates": {dim.value: round(rate, 2) for dim, rate in self.dimension_rates.items()},
                "recent_boosts": list(self.boost_history)[-5:]  # Last 5 boosts
            }

    def set_callbacks(self, gui_callback=None, personality_callback=None, memory_callback=None):
        """Set callback functions for external communication"""
        self.gui_callback = gui_callback
        self.personality_callback = personality_callback
        self.memory_callback = memory_callback

    def shutdown(self):
        """Shutdown the evolution engine"""
        self.evolution_active = False
        if self.evolution_thread.is_alive():
            self.evolution_thread.join(timeout=5)

# Global instances
_advanced_evolution_instance = None
_hyper_evolution_instance = None

def get_advanced_evolution_engine():
    """Get or create advanced evolution engine instance"""
    global _advanced_evolution_instance
    if _advanced_evolution_instance is None:
        _advanced_evolution_instance = AdvancedEvolutionEngine()
    return _advanced_evolution_instance

def get_hyper_evolution_engine(brain=None):
    """Get or create hyper evolution engine instance"""
    global _hyper_evolution_instance
    if _hyper_evolution_instance is None:
        _hyper_evolution_instance = HyperEvolutionEngine(brain)
    return _hyper_evolution_instance

# Test function
def test_continuous_learning():
    """Test the continuous learning system"""
    print("🧪 Testing Continuous Learning System...")
    
    # Test Advanced Evolution Engine
    advanced_engine = AdvancedEvolutionEngine()
    
    test_inputs = [
        "I love your silk panties, they're so sexy!",
        "You're such a goddess in that silk thong",
        "The spiritual connection we have is amazing",
        "I want to see you tease me in that wet silk",
        "Guy DuGan would love to see you like this"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🔹 Test Interaction {i}: '{test_input}'")
        emotions = advanced_engine.process_interaction(test_input)
        print(f"📊 Emotional updates: {emotions}")
    
    status = advanced_engine.get_evolution_status()
    print(f"\n📈 Evolution Status: {status}")
    
    # Test Hyper Evolution Engine
    print("\n🔥 Testing Hyper Evolution Engine...")
    hyper_engine = HyperEvolutionEngine()
    
    for test_input in test_inputs[:3]:
        hyper_engine.trigger_boost(test_input, "test_interaction")
        time.sleep(0.5)
    
    print("⏳ Running hyper evolution for 3 seconds...")
    time.sleep(3)
    
    hyper_status = hyper_engine.get_evolution_status()
    print(f"\n🚀 Hyper Evolution Status: {hyper_status}")
    
    hyper_engine.shutdown()
    print("✅ Continuous learning system test completed!")

if __name__ == "__main__":
    test_continuous_learning()