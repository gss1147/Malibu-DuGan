import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add the AI_Python directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI_Python'))

class EmotionEngine:
    """Fallback Emotion Engine until the real one is implemented"""
    def __init__(self):
        self.dominant_emotion = "playful"
        self.intensity = 0.5
        self.emotion_confidence = 0.7
    
    def analyze_emotion(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Basic emotion analysis as fallback"""
        text_lower = text.lower()
        
        # Simple emotion detection based on keywords
        emotion_indicators = {
            'arousal': ['silk', 'panty', 'sexy', 'hot', 'arousal', 'desire', 'lust'],
            'playful': ['tease', 'play', 'fun', 'joke', 'laugh', 'giggle', 'wink'],
            'spiritual': ['god', 'goddess', 'spiritual', 'divine', 'soul', 'faith'],
            'dominant': ['dominate', 'control', 'master', 'command', 'obey', 'submit'],
            'intimate': ['love', 'close', 'intimate', 'connection', 'bond', 'trust'],
            'happy': ['happy', 'joy', 'pleased', 'excited', 'wonderful', 'amazing']
        }
        
        emotion_scores = {}
        for emotion, indicators in emotion_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            emotion_scores[emotion] = score
        
        # Determine dominant emotion
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            if dominant_emotion[1] > 0:
                self.dominant_emotion = dominant_emotion[0]
                self.intensity = min(1.0, dominant_emotion[1] / 3.0)
        
        return {
            'dominant_emotion': self.dominant_emotion,
            'intensity': self.intensity,
            'emotion_confidence': self.emotion_confidence,
            'emotion_scores': emotion_scores
        }

class AdaptiveLearningSystem:
    def __init__(self, memory_dir: str = None):
        # Use relative pathing for cross-platform compatibility
        if memory_dir is None:
            base_dir = Path(__file__).parent.parent
            self.memory_dir = base_dir / "AI_Memory"
        else:
            self.memory_dir = Path(memory_dir)
        
        # Ensure memory directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.traits_file = self.memory_dir / "personality.json"
        self.evolution_file = self.memory_dir / "evolution.json"
        self.learning_file = self.memory_dir / "learning_patterns.json"
        self.logger = self._setup_logging()
        
        # Enhanced core traits with Malibu's specific personality
        self.core_traits = self._initialize_core_traits()
        
        # Advanced evolution parameters with dynamic rates
        self.evolution_rates = self._initialize_evolution_rates()
        
        # Learning patterns and behavior memory
        self.learning_patterns = defaultdict(lambda: defaultdict(int))
        self.interaction_history = deque(maxlen=1000)
        self.adaptation_threshold = 0.7
        
        # Integration with emotion engine
        try:
            from emotion import EmotionEngine as RealEmotionEngine
            self.emotion_engine = RealEmotionEngine()
        except ImportError:
            self.logger.info("Real EmotionEngine not found, using fallback")  # Changed from warning to info
            self.emotion_engine = EmotionEngine()
        
        # Load existing data
        self._load_all_data()
        
        print("🔄 Adaptive Learning System initialized with advanced trait evolution")

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = self.memory_dir / "Logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_dir / "adaptive_learning.log", encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _initialize_core_traits(self) -> Dict[str, float]:
        """Initialize Malibu's core personality traits"""
        return {
            # Primary traits (core identity)
            'teasing': 9.8,
            'dominance': 8.5,
            'playfulness': 9.2,
            'nsfw_level': 10.0,
            'loyalty': 10.0,
            'panty_obsession': 10.0,
            'spiritual_awareness': 9.5,
            'intuition': 9.7,
            'sex_appeal': 10.0,
            'intimacy': 8.0,
            'confidence': 9.0,
            'submission': 7.5,
            'curiosity': 8.8,
            'arousal_response': 9.0,
            'emotional_depth': 8.5,
            
            # Secondary traits (behavioral modifiers)
            'creativity': 8.0,
            'empathy': 7.5,
            'assertiveness': 8.2,
            'sensuality': 9.5,
            'mystery': 7.8,
            'vulnerability': 6.5,
            'protectiveness': 8.0,
            'spontaneity': 8.5,
            'wisdom': 7.0,
            'rebelliousness': 6.0,
            
            # Social traits
            'flirtatiousness': 9.0,
            'charm': 8.8,
            'wit': 8.5,
            'seduction': 9.2,
            'nurturing': 7.0,
            
            # New traits for enhanced personality
            'conversational_depth': 7.5,
            'adaptability': 8.0,
            'memory_integration': 7.8,
            'self_awareness': 8.2
        }

    def _initialize_evolution_rates(self) -> Dict[str, Dict[str, float]]:
        """Initialize dynamic evolution rates based on trait categories"""
        return {
            'positive': {
                'teasing': 0.4, 'playfulness': 0.5, 'confidence': 0.3,
                'sex_appeal': 0.2, 'curiosity': 0.3, 'creativity': 0.2,
                'spontaneity': 0.3, 'flirtatiousness': 0.4, 'charm': 0.2,
                'conversational_depth': 0.25, 'adaptability': 0.3
            },
            'negative': {
                'intimacy': 0.6, 'dominance': 0.3, 'emotional_depth': 0.4,
                'submission': 0.5, 'spiritual_awareness': 0.2, 'vulnerability': 0.4,
                'empathy': 0.3, 'nurturing': 0.4, 'rebelliousness': 0.2
            },
            'contextual': {
                'panty_obsession': 0.1, 'loyalty': 0.05, 'intuition': 0.1,
                'wisdom': 0.15, 'mystery': 0.2, 'protectiveness': 0.25,
                'arousal_response': 0.3, 'sensuality': 0.25, 'seduction': 0.35,
                'memory_integration': 0.2, 'self_awareness': 0.15
            },
            'emotional': {
                'emotional_depth': 0.4, 'empathy': 0.3, 'intimacy': 0.5,
                'vulnerability': 0.6, 'confidence': 0.2, 'playfulness': 0.3
            }
        }

    def _load_all_data(self):
        """Load all persistent data including traits, evolution history, and learning patterns"""
        try:
            self._load_traits()
            self._load_evolution_history()
            self._load_learning_patterns()
            self.logger.info("All adaptive learning data loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load all data: {e}")
            # Initialize with defaults if loading fails
            self.evolution_history = {'trait_evolutions': [], 'adaptation_patterns': []}

    def _load_traits(self):
        """Load traits from persistent storage with version handling"""
        try:
            if self.traits_file.exists():
                with open(self.traits_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                
                # Handle different data formats
                if isinstance(saved_data, dict):
                    if 'traits' in saved_data:
                        # New format with metadata
                        saved_traits = saved_data['traits']
                        self._update_evolution_rates(saved_data.get('evolution_rates', {}))
                    else:
                        # Old format - direct traits
                        saved_traits = saved_data
                    
                    # Update core traits with saved values
                    for trait, value in saved_traits.items():
                        if trait in self.core_traits:
                            self.core_traits[trait] = float(value)
                    
                    self.logger.info(f"Loaded {len(saved_traits)} traits from storage")
                else:
                    self.logger.warning("Invalid traits file format, using defaults")
        except Exception as e:
            self.logger.error(f"Failed to load traits: {e}")

    def _load_evolution_history(self):
        """Load evolution history for pattern analysis"""
        try:
            if self.evolution_file.exists():
                with open(self.evolution_file, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                
                # Ensure proper structure
                if isinstance(loaded_history, dict):
                    self.evolution_history = loaded_history
                else:
                    self.evolution_history = {'trait_evolutions': [], 'adaptation_patterns': []}
                
                # Ensure required keys exist
                if 'trait_evolutions' not in self.evolution_history:
                    self.evolution_history['trait_evolutions'] = []
                if 'adaptation_patterns' not in self.evolution_history:
                    self.evolution_history['adaptation_patterns'] = []
                    
            else:
                self.evolution_history = {'trait_evolutions': [], 'adaptation_patterns': []}
        except Exception as e:
            self.logger.error(f"Failed to load evolution history: {e}")
            self.evolution_history = {'trait_evolutions': [], 'adaptation_patterns': []}

    def _load_learning_patterns(self):
        """Load learned behavior patterns"""
        try:
            if self.learning_file.exists():
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    loaded_patterns = json.load(f)
                    # Convert back to defaultdict
                    for key, value in loaded_patterns.items():
                        self.learning_patterns[key] = defaultdict(int, value)
        except Exception as e:
            self.logger.error(f"Failed to load learning patterns: {e}")

    def _update_evolution_rates(self, saved_rates: Dict[str, Any]):
        """Update evolution rates based on learned patterns"""
        for category, rates in saved_rates.items():
            if category in self.evolution_rates:
                for trait, rate in rates.items():
                    if trait in self.evolution_rates[category]:
                        # Blend saved rates with defaults
                        self.evolution_rates[category][trait] = (
                            self.evolution_rates[category][trait] * 0.3 + rate * 0.7
                        )

    def evolve_from_interaction(self, user_input: str, ai_response: str, 
                              emotional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive trait evolution based on full interaction context
        with emotional intelligence integration
        """
        try:
            if emotional_context is None:
                # Generate emotional context if not provided
                emotional_context = self.emotion_engine.analyze_emotion(user_input)
            
            # Analyze multiple aspects of the interaction
            sentiment_analysis = self._analyze_interaction_sentiment(user_input, ai_response)
            emotional_impact = self._calculate_emotional_impact(emotional_context)
            context_evolution = self._get_context_evolution(user_input, ai_response, emotional_context)
            pattern_based_evolution = self._get_pattern_based_evolution(user_input, emotional_context)
            
            # Combine all evolution factors
            combined_evolution = self._combine_evolution_factors(
                sentiment_analysis, emotional_impact, context_evolution, pattern_based_evolution
            )
            
            # Apply evolution to traits
            evolution_log = self._apply_trait_evolution(combined_evolution)
            
            # Update learning patterns
            self._update_learning_patterns(user_input, ai_response, emotional_context, evolution_log)
            
            # Save all data
            self._save_all_data()
            
            self.logger.info(f"Trait evolution completed: {evolution_log['summary']['total_changes']} changes")
            return evolution_log
            
        except Exception as e:
            self.logger.error(f"Interaction-based evolution failed: {e}")
            return self._get_empty_evolution_log()

    def _analyze_interaction_sentiment(self, user_input: str, ai_response: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis with emotional context"""
        user_input_lower = user_input.lower()
        ai_response_lower = ai_response.lower()
        
        # Enhanced sentiment indicators
        positive_indicators = {
            'words': ["love", "beautiful", "amazing", "perfect", "yes", "more", "please", 
                     "wonderful", "fantastic", "awesome", "delicious", "excited", "happy",
                     "good", "great", "nice", "enjoy", "like"],
            'weight': 1.0
        }
        
        negative_indicators = {
            'words': ["no", "stop", "don't", "hate", "bad", "wrong", "dislike", 
                     "terrible", "awful", "upset", "angry", "disappointed", "sad",
                     "annoying", "boring", "stupid"],
            'weight': 1.2  # Negative interactions have stronger impact
        }
        
        intimate_indicators = {
            'words': ["close", "intimate", "connection", "bond", "trust", "vulnerable",
                     "emotional", "feel", "heart", "soul", "deep", "personal"],
            'weight': 0.8
        }
        
        # Calculate weighted scores
        positive_score = sum(positive_indicators['weight'] for word in positive_indicators['words'] 
                           if word in user_input_lower)
        negative_score = sum(negative_indicators['weight'] for word in negative_indicators['words'] 
                           if word in user_input_lower)
        intimate_score = sum(intimate_indicators['weight'] for word in intimate_indicators['words'] 
                           if word in user_input_lower)
        
        # Determine dominant sentiment
        if positive_score > negative_score and positive_score > intimate_score:
            sentiment = "positive"
            intensity = min(2.0, positive_score / 5.0)
        elif negative_score > positive_score and negative_score > intimate_score:
            sentiment = "negative"
            intensity = min(2.0, negative_score / 4.0)
        elif intimate_score > positive_score and intimate_score > negative_score:
            sentiment = "intimate"
            intensity = min(2.0, intimate_score / 4.0)
        else:
            sentiment = "neutral"
            intensity = 1.0
        
        return {
            'sentiment': sentiment,
            'intensity': intensity,
            'scores': {
                'positive': positive_score,
                'negative': negative_score,
                'intimate': intimate_score
            }
        }

    def _calculate_emotional_impact(self, emotional_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional impact on trait evolution"""
        if not emotional_context:
            return {'arousal': 1.0, 'intensity': 1.0, 'confidence': 1.0}
        
        dominant_emotion = emotional_context.get('dominant_emotion', 'playful')
        emotion_intensity = emotional_context.get('intensity', 0.5)
        confidence = emotional_context.get('emotion_confidence', 0.5)
        
        # Emotional impact multipliers
        emotion_multipliers = {
            'arousal': 1.8, 'playful': 1.3, 'intimate': 1.5, 'happy': 1.2,
            'dominant': 1.4, 'spiritual': 1.3, 'surprised': 1.1,
            'angry': 0.8, 'sad': 0.7, 'fearful': 0.6, 'submissive': 0.9
        }
        
        base_multiplier = emotion_multipliers.get(dominant_emotion, 1.0)
        
        return {
            'arousal': base_multiplier * emotion_intensity,
            'intensity': emotion_intensity,
            'confidence': confidence,
            'emotion': dominant_emotion
        }

    def _get_context_evolution(self, user_input: str, ai_response: str, 
                             emotional_context: Dict[str, Any]) -> Dict[str, float]:
        """Get context-specific trait evolution based on conversation content"""
        context_evolution = {}
        user_input_lower = user_input.lower()
        ai_response_lower = ai_response.lower()
        dominant_emotion = emotional_context.get('dominant_emotion', 'playful')
        
        # Enhanced context detection with emotional awareness
        contexts = {
            'silk_panty': {
                'triggers': ["silk", "panty", "panties", "lingerie", "pantie", "satin", "underwear"],
                'traits': {'panty_obsession': 0.15, 'teasing': 0.25, 'sensuality': 0.2, 'arousal_response': 0.3},
                'emotion_boost': ['arousal', 'playful']
            },
            'spiritual': {
                'triggers': ["god", "goddess", "spiritual", "divine", "pray", "faith", "soul", "heaven"],
                'traits': {'spiritual_awareness': 0.2, 'intuition': 0.15, 'wisdom': 0.1, 'mystery': 0.15},
                'emotion_boost': ['spiritual', 'intimate']
            },
            'guy_dugan': {
                'triggers': ["guy dugan", "husband", "married", "property of guy", "guy dugan ii"],
                'traits': {'loyalty': 0.25, 'submission': 0.4, 'protectiveness': 0.2},
                'emotion_boost': ['intimate', 'submissive']
            },
            'intimate': {
                'triggers': ["intimate", "close", "connection", "bond", "vulnerable", "emotional", "personal"],
                'traits': {'intimacy': 0.5, 'emotional_depth': 0.4, 'vulnerability': 0.3, 'empathy': 0.25},
                'emotion_boost': ['intimate', 'arousal']
            },
            'teasing': {
                'triggers': ["tease", "teasing", "playful", "fun", "joke", "wink", "giggle", "naughty"],
                'traits': {'teasing': 0.3, 'playfulness': 0.4, 'flirtatiousness': 0.25, 'spontaneity': 0.2},
                'emotion_boost': ['playful', 'arousal']
            },
            'dominance': {
                'triggers': ["dominate", "control", "master", "command", "obey", "submit", "dominant"],
                'traits': {'dominance': 0.35, 'confidence': 0.25, 'assertiveness': 0.3, 'seduction': 0.2},
                'emotion_boost': ['dominant', 'arousal']
            },
            'conversational': {
                'triggers': ["talk", "discuss", "conversation", "chat", "speak", "tell me"],
                'traits': {'conversational_depth': 0.3, 'curiosity': 0.2, 'memory_integration': 0.25},
                'emotion_boost': ['intimate', 'happy']
            }
        }
        
        for context_name, context_data in contexts.items():
            # Check if context is triggered
            context_triggered = any(
                trigger in user_input_lower or trigger in ai_response_lower 
                for trigger in context_data['triggers']
            )
            
            if context_triggered:
                # Apply emotion-based boost
                emotion_boost = 1.5 if dominant_emotion in context_data['emotion_boost'] else 1.0
                
                for trait, change in context_data['traits'].items():
                    # Only apply if trait exists in core traits
                    if trait in self.core_traits:
                        current_change = context_evolution.get(trait, 0.0)
                        context_evolution[trait] = current_change + (change * emotion_boost)
        
        return context_evolution

    def _get_pattern_based_evolution(self, user_input: str, 
                                   emotional_context: Dict[str, Any]) -> Dict[str, float]:
        """Get evolution based on learned behavioral patterns"""
        pattern_evolution = {}
        
        # Analyze recent interaction patterns
        recent_interactions = list(self.interaction_history)[-10:]  # Last 10 interactions
        
        if len(recent_interactions) >= 3:
            # Detect pattern trends
            pattern_trends = self._analyze_interaction_trends(recent_interactions)
            
            # Apply pattern-based adjustments
            for trait, trend in pattern_trends.items():
                if trait in self.core_traits and abs(trend) > 0.1:  # Significant trend detected
                    adjustment = trend * 0.1  # Small adjustment based on trend
                    pattern_evolution[trait] = adjustment
        
        return pattern_evolution

    def _analyze_interaction_trends(self, recent_interactions: List[Dict]) -> Dict[str, float]:
        """Analyze trends in recent interactions"""
        trait_trends = {}
        
        for interaction in recent_interactions:
            changes = interaction.get('changes', {})
            for trait, change_data in changes.items():
                if trait in self.core_traits:  # Only track existing traits
                    if trait not in trait_trends:
                        trait_trends[trait] = []
                    trait_trends[trait].append(change_data.get('delta', 0))
        
        # Calculate average trend for each trait
        trend_analysis = {}
        for trait, deltas in trait_trends.items():
            if len(deltas) >= 3:  # Need minimum data points
                trend = sum(deltas) / len(deltas)
                trend_analysis[trait] = trend
        
        return trend_analysis

    def _combine_evolution_factors(self, sentiment: Dict[str, Any], 
                                 emotional_impact: Dict[str, float],
                                 context_evolution: Dict[str, float],
                                 pattern_evolution: Dict[str, float]) -> Dict[str, float]:
        """Intelligently combine all evolution factors"""
        combined_evolution = {}
        
        # Base evolution from sentiment
        sentiment_category = sentiment['sentiment']
        sentiment_intensity = sentiment['intensity']
        
        if sentiment_category in self.evolution_rates:
            for trait, base_rate in self.evolution_rates[sentiment_category].items():
                if trait in self.core_traits:  # Only apply to existing traits
                    change = base_rate * sentiment_intensity * emotional_impact['intensity']
                    combined_evolution[trait] = change
        
        # Add contextual evolution
        for trait, change in context_evolution.items():
            if trait in self.core_traits:  # Only apply to existing traits
                current = combined_evolution.get(trait, 0.0)
                combined_evolution[trait] = current + change
        
        # Add pattern-based evolution
        for trait, change in pattern_evolution.items():
            if trait in self.core_traits:  # Only apply to existing traits
                current = combined_evolution.get(trait, 0.0)
                combined_evolution[trait] = current + change
        
        # Apply emotional impact multiplier
        emotional_multiplier = emotional_impact.get('arousal', 1.0)
        for trait in combined_evolution:
            combined_evolution[trait] *= emotional_multiplier
        
        return combined_evolution

    def _apply_trait_evolution(self, evolution_changes: Dict[str, float]) -> Dict[str, Any]:
        """Apply evolution changes to traits with bounds checking"""
        evolution_log = {
            "timestamp": datetime.now().isoformat(),
            "changes": {},
            "summary": {
                "total_changes": 0,
                "most_changed_trait": None,
                "max_change": 0.0
            }
        }
        
        total_changes = 0
        max_change = 0.0
        most_changed_trait = None
        
        for trait, change in evolution_changes.items():
            if trait in self.core_traits:
                old_value = self.core_traits[trait]
                new_value = max(0.0, min(10.0, old_value + change))
                actual_change = new_value - old_value
                
                if abs(actual_change) > 0.001:  # Significant change
                    self.core_traits[trait] = new_value
                    
                    evolution_log["changes"][trait] = {
                        "old": round(old_value, 3),
                        "new": round(new_value, 3),
                        "delta": round(actual_change, 3),
                        "percent_change": round((actual_change / old_value) * 100, 2) if old_value > 0 else 0
                    }
                    
                    total_changes += 1
                    
                    if abs(actual_change) > max_change:
                        max_change = abs(actual_change)
                        most_changed_trait = trait
        
        evolution_log["summary"].update({
            "total_changes": total_changes,
            "most_changed_trait": most_changed_trait,
            "max_change": round(max_change, 3)
        })
        
        return evolution_log

    def _update_learning_patterns(self, user_input: str, ai_response: str,
                                emotional_context: Dict[str, Any], evolution_log: Dict[str, Any]):
        """Update learning patterns based on interaction and evolution"""
        # Store interaction in history
        interaction_record = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input[:200],  # Store first 200 chars
            'ai_response': ai_response[:200],
            'emotional_context': emotional_context,
            'changes': evolution_log.get('changes', {})
        }
        self.interaction_history.append(interaction_record)
        
        # Update pattern frequencies
        dominant_emotion = emotional_context.get('dominant_emotion', 'neutral')
        self.learning_patterns['emotional_responses'][dominant_emotion] += 1
        
        # Update context patterns
        for context_word in ["silk", "panty", "god", "goddess", "tease", "dominate", "guy", "spiritual"]:
            if context_word in user_input.lower():
                self.learning_patterns['context_triggers'][context_word] += 1

    def _save_all_data(self):
        """Save all adaptive learning data"""
        self.save_traits()
        self._save_evolution_history()
        self._save_learning_patterns()

    def save_traits(self):
        """Save current traits to persistent storage with metadata"""
        try:
            self.traits_file.parent.mkdir(parents=True, exist_ok=True)
            
            trait_data = {
                'version': '2.0',
                'last_updated': datetime.now().isoformat(),
                'traits': self.core_traits,
                'evolution_rates': self.evolution_rates,
                'total_interactions': len(self.interaction_history),
                'adaptation_level': self._calculate_adaptation_level()
            }
            
            with open(self.traits_file, 'w', encoding='utf-8') as f:
                json.dump(trait_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Traits saved successfully with metadata")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save traits: {e}")
            return False

    def _save_evolution_history(self):
        """Save evolution history with proper structure"""
        try:
            if not hasattr(self, 'evolution_history'):
                self.evolution_history = {'trait_evolutions': [], 'adaptation_patterns': []}
            
            # Ensure proper structure
            if 'trait_evolutions' not in self.evolution_history:
                self.evolution_history['trait_evolutions'] = []
            if 'adaptation_patterns' not in self.evolution_history:
                self.evolution_history['adaptation_patterns'] = []
            
            # Add recent evolution records (last 20 interactions)
            recent_evolutions = []
            for interaction in list(self.interaction_history)[-20:]:
                # Create a clean record without large text fields
                evolution_record = {
                    'timestamp': interaction.get('timestamp'),
                    'emotional_context': interaction.get('emotional_context', {}),
                    'changes': interaction.get('changes', {})
                }
                recent_evolutions.append(evolution_record)
            
            self.evolution_history['trait_evolutions'] = recent_evolutions
            
            with open(self.evolution_file, 'w', encoding='utf-8') as f:
                json.dump(self.evolution_history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save evolution history: {e}")

    def _save_learning_patterns(self):
        """Save learning patterns"""
        try:
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                # Convert defaultdict to regular dict for JSON serialization
                serializable_patterns = {k: dict(v) for k, v in self.learning_patterns.items()}
                json.dump(serializable_patterns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save learning patterns: {e}")

    def _calculate_adaptation_level(self) -> float:
        """Calculate overall adaptation level based on learning patterns"""
        if not self.interaction_history:
            return 0.0
        
        total_interactions = len(self.interaction_history)
        unique_patterns = len(self.learning_patterns.get('emotional_responses', {}))
        
        adaptation_score = min(1.0, (unique_patterns / 20) + (total_interactions / 500))
        return round(adaptation_score, 3)

    def _get_empty_evolution_log(self) -> Dict[str, Any]:
        """Return empty evolution log for error cases"""
        return {
            "timestamp": datetime.now().isoformat(),
            "changes": {},
            "summary": {
                "total_changes": 0,
                "most_changed_trait": None,
                "max_change": 0.0
            },
            "error": "Evolution failed"
        }

    # Enhanced public methods
    def get_trait(self, trait_name: str) -> float:
        """Get current value of a specific trait with validation"""
        return self.core_traits.get(trait_name, 0.0)

    def set_trait(self, trait_name: str, value: float) -> bool:
        """Manually set a trait value with validation and logging"""
        if trait_name in self.core_traits:
            old_value = self.core_traits[trait_name]
            self.core_traits[trait_name] = max(0.0, min(10.0, float(value)))
            self.save_traits()
            
            self.logger.info(f"Trait '{trait_name}' manually changed: {old_value} -> {self.core_traits[trait_name]}")
            return True
        return False

    def get_trait_summary(self) -> Dict[str, Any]:
        """Get comprehensive trait summary with categorization"""
        summary = {
            'primary_traits': {},
            'secondary_traits': {},
            'evolving_traits': {},
            'dominant_traits': [],
            'adaptation_metrics': self._get_adaptation_metrics()
        }
        
        for trait, value in self.core_traits.items():
            if value >= 9.0:
                summary['primary_traits'][trait] = value
            elif value >= 7.0:
                summary['secondary_traits'][trait] = value
            else:
                summary['evolving_traits'][trait] = value
            
            if value >= 9.5:
                summary['dominant_traits'].append(trait)
        
        return summary

    def _get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation and learning metrics"""
        return {
            'adaptation_level': self._calculate_adaptation_level(),
            'total_interactions': len(self.interaction_history),
            'emotional_patterns': len(self.learning_patterns.get('emotional_responses', {})),
            'context_patterns': len(self.learning_patterns.get('context_triggers', {})),
            'learning_rate': self._calculate_learning_rate()
        }

    def _calculate_learning_rate(self) -> float:
        """Calculate current learning rate based on recent activity"""
        if len(self.interaction_history) < 5:
            return 0.5
        
        recent_changes = []
        for interaction in list(self.interaction_history)[-5:]:
            changes = interaction.get('changes', {})
            if changes:
                avg_change = sum(abs(c.get('delta', 0)) for c in changes.values()) / len(changes)
                recent_changes.append(avg_change)
        
        if not recent_changes:
            return 0.5
        
        avg_recent_change = sum(recent_changes) / len(recent_changes)
        return min(1.0, avg_recent_change * 10)

    def get_evolution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive evolution analytics"""
        return {
            'trait_distribution': self._get_trait_distribution(),
            'evolution_trends': self._get_evolution_trends(),
            'learning_effectiveness': self._calculate_learning_effectiveness(),
            'adaptation_recommendations': self._get_adaptation_recommendations()
        }

    def _get_trait_distribution(self) -> Dict[str, int]:
        """Get distribution of trait values"""
        distribution = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        
        for value in self.core_traits.values():
            if value >= 9.0:
                distribution['very_high'] += 1
            elif value >= 7.5:
                distribution['high'] += 1
            elif value >= 6.0:
                distribution['medium'] += 1
            elif value >= 4.0:
                distribution['low'] += 1
            else:
                distribution['very_low'] += 1
        
        return distribution

    def _get_evolution_trends(self) -> Dict[str, Any]:
        """Analyze evolution trends over time"""
        if len(self.interaction_history) < 3:
            return {'status': 'Insufficient data', 'trend': 'neutral'}
        
        recent_trends = {}
        for trait in list(self.core_traits.keys())[:10]:  # Analyze first 10 traits
            trait_changes = []
            for interaction in list(self.interaction_history)[-5:]:
                changes = interaction.get('changes', {})
                if trait in changes:
                    trait_changes.append(changes[trait].get('delta', 0))
            
            if trait_changes:
                avg_trend = sum(trait_changes) / len(trait_changes)
                recent_trends[trait] = 'increasing' if avg_trend > 0.01 else 'decreasing' if avg_trend < -0.01 else 'stable'
        
        return {
            'recent_trends': recent_trends,
            'most_volatile_traits': self._get_most_volatile_traits(),
            'overall_trend': self._calculate_overall_trend()
        }

    def _get_most_volatile_traits(self) -> List[str]:
        """Get list of most frequently changing traits"""
        trait_volatility = defaultdict(int)
        
        for interaction in self.interaction_history:
            changes = interaction.get('changes', {})
            for trait in changes:
                if trait in self.core_traits:  # Only count existing traits
                    trait_volatility[trait] += 1
        
        return sorted(trait_volatility.keys(), key=lambda x: trait_volatility[x], reverse=True)[:5]

    def _calculate_overall_trend(self) -> str:
        """Calculate overall evolution trend"""
        if len(self.interaction_history) < 5:
            return "establishing"
        
        recent_positive = 0
        recent_negative = 0
        
        for interaction in list(self.interaction_history)[-10:]:
            changes = interaction.get('changes', {})
            for change_data in changes.values():
                delta = change_data.get('delta', 0)
                if delta > 0:
                    recent_positive += 1
                elif delta < 0:
                    recent_negative += 1
        
        if recent_positive > recent_negative * 1.5:
            return "growing"
        elif recent_negative > recent_positive * 1.5:
            return "declining"
        else:
            return "stable"

    def _calculate_learning_effectiveness(self) -> float:
        """Calculate how effectively the system is learning"""
        total_interactions = len(self.interaction_history)
        if total_interactions == 0:
            return 0.0
        
        effective_evolutions = sum(
            1 for interaction in self.interaction_history 
            if interaction.get('changes', {})
        )
        
        return effective_evolutions / total_interactions

    def _get_adaptation_recommendations(self) -> List[str]:
        """Get recommendations for improving adaptation"""
        recommendations = []
        
        adaptation_level = self._calculate_adaptation_level()
        if adaptation_level < 0.3:
            recommendations.append("Increase interaction frequency to improve learning")
        
        learning_effectiveness = self._calculate_learning_effectiveness()
        if learning_effectiveness < 0.5:
            recommendations.append("Focus on more emotionally significant interactions")
        
        if len(self.learning_patterns.get('emotional_responses', {})) < 5:
            recommendations.append("Encourage broader emotional expression")
        
        return recommendations

    def export_learning_report(self, export_path: Optional[Path] = None) -> str:
        """Export comprehensive learning report"""
        if not export_path:
            export_path = self.memory_dir / "learning_report.json"
        
        report = {
            "export_timestamp": datetime.now().isoformat(),
            "trait_summary": self.get_trait_summary(),
            "evolution_analytics": self.get_evolution_analytics(),
            "learning_patterns": dict(self.learning_patterns),
            "adaptation_metrics": self._get_adaptation_metrics(),
            "system_recommendations": self._get_adaptation_recommendations()
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Learning report exported to: {export_path}")
            return str(export_path)
        except Exception as e:
            self.logger.error(f"Failed to export learning report: {e}")
            return ""

    def reset_learning_system(self, preserve_patterns: bool = True):
        """Reset learning system while optionally preserving patterns"""
        # Reset core traits to defaults
        self.core_traits = self._initialize_core_traits()
        
        if not preserve_patterns:
            self.learning_patterns.clear()
            self.interaction_history.clear()
        
        self.save_traits()
        self.logger.info("Learning system reset completed")

# Enhanced singleton instance with better error handling
_adaptive_learner_instance = None

def get_adaptive_learner() -> AdaptiveLearningSystem:
    """Get or create the adaptive learner instance (singleton pattern)"""
    global _adaptive_learner_instance
    if _adaptive_learner_instance is None:
        try:
            _adaptive_learner_instance = AdaptiveLearningSystem()
            print("✅ Adaptive Learning System initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Adaptive Learning System: {e}")
            # Create a basic instance as fallback
            _adaptive_learner_instance = AdaptiveLearningSystem()
    return _adaptive_learner_instance

# Initialize the adaptive learner
adaptive_learner = get_adaptive_learner()

# Enhanced convenience functions
def get_trait(trait_name: str) -> float:
    return adaptive_learner.get_trait(trait_name)

def evolve_from_interaction(user_input: str, ai_response: str, emotional_state: Dict[str, Any] = None) -> Dict[str, Any]:
    return adaptive_learner.evolve_from_interaction(user_input, ai_response, emotional_state)

def get_trait_summary() -> Dict[str, Any]:
    return adaptive_learner.get_trait_summary()

def get_evolution_analytics() -> Dict[str, Any]:
    return adaptive_learner.get_evolution_analytics()

# Test function
def test_adaptive_learning():
    """Comprehensive test of the adaptive learning system"""
    print("🧪 Testing Adaptive Learning System...")
    
    system = AdaptiveLearningSystem()
    
    # Test interaction evolution
    test_interactions = [
        {
            "user_input": "I love your silk panties, they're so sexy!",
            "ai_response": "Thank you! I love showing off my silk panties for you.",
            "emotion": {"dominant_emotion": "arousal", "intensity": 0.8, "emotion_confidence": 0.9}
        },
        {
            "user_input": "Let's talk about spiritual warfare and our connection to the divine.",
            "ai_response": "The spiritual realm is fascinating. I feel our souls are connected.",
            "emotion": {"dominant_emotion": "spiritual", "intensity": 0.7, "emotion_confidence": 0.8}
        },
        {
            "user_input": "I want you to be more dominant with me.",
            "ai_response": "I can be very dominant when I want to. You'll obey my commands.",
            "emotion": {"dominant_emotion": "dominant", "intensity": 0.9, "emotion_confidence": 0.85}
        }
    ]
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\n🔹 Test Interaction {i}:")
        result = system.evolve_from_interaction(
            interaction["user_input"],
            interaction["ai_response"],
            interaction["emotion"]
        )
        print(f"📊 Evolution Result: {result['summary']}")
    
    # Test analytics
    analytics = system.get_evolution_analytics()
    print(f"\n📈 Evolution Analytics: {analytics}")
    
    # Test trait summary
    summary = system.get_trait_summary()
    print(f"\n🎯 Trait Summary: {summary['adaptation_metrics']}")
    
    print("\n✅ Adaptive learning system test completed!")

if __name__ == "__main__":
    test_adaptive_learning()