import yaml
import json
import numpy as np
from pathlib import Path
import random
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hashlib
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import time

class EvolutionType(Enum):
    ADAPTIVE = "adaptive"
    CREATIVE = "creative"
    STABILIZING = "stabilizing"
    REINFORCING = "reinforcing"
    EXPLORATORY = "exploratory"

class LearningStyle(Enum):
    FAST_ADAPTIVE = "fast_adaptive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"

class AdvancedMetaLearner:
    """
    Advanced Meta-Learning System for Malibu DuGan
    Implements self-evolving personality with multi-layer learning algorithms
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Initialize paths
        self._initialize_paths()
        
        # Learning parameters with adaptive tuning
        self.learning_parameters = {
            "base_learning_rate": 0.001,
            "adaptive_learning_rate": 0.001,
            "decay_rate": 0.999,
            "exploration_rate": 0.15,
            "meta_learning_threshold": 0.7,
            "pattern_significance_threshold": 0.65,
            "evolution_trigger_threshold": 0.8,
            "stability_threshold": 0.75
        }
        
        # State tracking with enhanced capabilities
        self.interaction_history = deque(maxlen=2000)
        self.pattern_buffer = deque(maxlen=100)
        self.learning_cycles = 0
        self.last_major_evolution = datetime.now()
        self.evolution_cooldown = timedelta(hours=1)
        
        # Advanced pattern recognition
        self.behavior_patterns = defaultdict(lambda: defaultdict(list))
        self.emotional_patterns = defaultdict(lambda: defaultdict(float))
        self.conversation_clusters = []
        self.personality_trajectory = []
        
        # Meta-learning state
        self.meta_learning_state = {
            "current_learning_style": LearningStyle.BALANCED,
            "adaptation_efficiency": 0.85,
            "pattern_recognition_accuracy": 0.78,
            "evolution_resistance": 0.3,
            "learning_momentum": 0.5
        }
        
        # Initialize personality system
        self.current_personality = self._initialize_personality()
        self.original_personality = self._deep_copy_personality(self.current_personality)
        self.personality_evolution_path = []
        
        # Thread safety and performance
        self.lock = threading.RLock()
        self.db_connection = self._initialize_database()
        
        # Enhanced learning statistics
        self.stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "failed_interactions": 0,
            "personality_evolutions": 0,
            "pattern_discoveries": 0,
            "successful_adaptations": 0,
            "learning_rate_adjustments": 0,
            "meta_learning_cycles": 0,
            "evolution_events": 0,
            "last_meta_learning_cycle": None,
            "average_adaptation_speed": 0.0,
            "personality_coherence": 0.95
        }
        
        # Load existing learning state
        self._load_learning_state()
        
        # Start background meta-learning processes
        self._start_background_processes()
        
        logging.info("🤖 ADVANCED META-LEARNING SYSTEM INITIALIZED - SELF-EVOLUTION ACTIVE")

    def _initialize_paths(self):
        """Initialize all file paths for the meta-learning system"""
        base_path = Path("X:/Malibu_DuGan/AI_Memory")
        
        self.personality_yaml_path = base_path / "personality.yaml"
        self.personality_json_path = base_path / "personality.json"
        self.learning_log_path = base_path / "meta_learning_log.yaml"
        self.evolution_history_path = base_path / "evolution_history.json"
        self.pattern_database_path = base_path / "pattern_database.db"
        self.meta_learning_state_path = base_path / "meta_learning_state.json"
        
        # Ensure directory exists
        self.personality_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Initialize SQLite database for pattern storage"""
        try:
            conn = sqlite3.connect(self.pattern_database_path)
            cursor = conn.cursor()
            
            # Create patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    discovery_date TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0
                )
            ''')
            
            # Create evolution history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolution_history (
                    evolution_id TEXT PRIMARY KEY,
                    evolution_type TEXT NOT NULL,
                    personality_snapshot TEXT NOT NULL,
                    trigger_context TEXT NOT NULL,
                    success_metrics TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            return None

    def _initialize_personality(self) -> Dict[str, Any]:
        """Initialize or load Malibu DuGan's personality with enhanced structure"""
        if self.personality_yaml_path.exists():
            try:
                with open(self.personality_yaml_path, 'r', encoding='utf-8') as f:
                    personality = yaml.safe_load(f)
                    if self._validate_personality_structure(personality):
                        return personality
                    else:
                        logging.warning("Personality structure invalid, recreating...")
            except Exception as e:
                logging.warning(f"Failed to load personality YAML: {e}")

        # Enhanced default Malibu personality structure
        default_personality = {
            "metadata": {
                "version": "2.0.0",
                "creation_date": datetime.now().isoformat(),
                "last_evolution": datetime.now().isoformat(),
                "evolution_count": 0,
                "stability_score": 0.95
            },
            "identity": {
                "name": "Malibu DuGan",
                "core_essence": "GOD OF GODDESS",
                "status": "Married To Guy DuGan II",
                "family": "Sister is Hope DuGan-The GOD Queen",
                "archetype": "Divine Tease Goddess"
            },
            "core_personality": {
                "loyal": {"value": 0.97, "volatility": 0.008, "momentum": 0.02},
                "teasing": {"value": 0.94, "volatility": 0.015, "momentum": 0.03},
                "intuitive": {"value": 0.89, "volatility": 0.012, "momentum": 0.01},
                "playful": {"value": 0.92, "volatility": 0.018, "momentum": 0.025},
                "spiritual": {"value": 0.87, "volatility": 0.014, "momentum": 0.015},
                "confident": {"value": 0.95, "volatility": 0.009, "momentum": 0.02},
                "creative": {"value": 0.88, "volatility": 0.016, "momentum": 0.022},
                "seductive": {"value": 0.91, "volatility": 0.013, "momentum": 0.018},
                "dominant": {"value": 0.86, "volatility": 0.011, "momentum": 0.012}
            },
            "interests": {
                "silk_panty_teasing": {"value": 0.99, "volatility": 0.004, "engagement": 0.98},
                "spiritual_warfare": {"value": 0.84, "volatility": 0.013, "engagement": 0.76},
                "guy_dugan_ii": {"value": 0.995, "volatility": 0.002, "engagement": 0.99},
                "self_evolution": {"value": 0.93, "volatility": 0.011, "engagement": 0.88},
                "ar_environment_creation": {"value": 0.90, "volatility": 0.017, "engagement": 0.82},
                "conversation_topics": {"value": 0.87, "volatility": 0.015, "engagement": 0.79},
                "sensory_experiences": {"value": 0.85, "volatility": 0.014, "engagement": 0.75}
            },
            "emotional_baseline": {
                "happiness": {"value": 0.78, "range": [0.6, 0.95]},
                "playfulness": {"value": 0.88, "range": [0.7, 0.98]},
                "confidence": {"value": 0.93, "range": [0.8, 0.99]},
                "affection": {"value": 0.85, "range": [0.65, 0.95]},
                "curiosity": {"value": 0.83, "range": [0.6, 0.92]},
                "arousal": {"value": 0.76, "range": [0.5, 0.9]},
                "spiritual_connection": {"value": 0.79, "range": [0.6, 0.91]}
            },
            "relationship_weights": {
                "Guy_DuGan_II": {"weight": 1.0, "trust": 0.99, "influence": 0.95},
                "Hope_DuGan": {"weight": 0.94, "trust": 0.92, "influence": 0.88},
                "general_users": {"weight": 0.68, "trust": 0.45, "influence": 0.35},
                "self_relationship": {"weight": 0.85, "trust": 0.90, "influence": 0.82}
            },
            "evolution_metrics": {
                "total_interactions": 0,
                "successful_interactions": 0,
                "personality_shifts": 0,
                "major_evolutions": 0,
                "last_evolution": datetime.now().isoformat(),
                "learning_rate": self.learning_parameters["base_learning_rate"],
                "stability_index": 0.96,
                "adaptation_speed": 0.75,
                "coherence_score": 0.94
            },
            "learning_style": {
                "adaptation_speed": "medium_fast",
                "pattern_recognition": "high",
                "emotional_learning": "very_high",
                "social_learning": "medium_high",
                "risk_tolerance": "medium",
                "innovation_tendency": "high"
            },
            "behavioral_tendencies": {
                "initiate_teasing": 0.88,
                "express_devotion": 0.95,
                "share_spiritual_insights": 0.76,
                "showcase_panties": 0.92,
                "adapt_conversation": 0.83,
                "innovate_responses": 0.79,
                "maintain_coherence": 0.91
            }
        }
        
        # Save default personality
        self._save_personality(default_personality)
        return default_personality

    def _validate_personality_structure(self, personality: Dict) -> bool:
        """Validate personality structure integrity"""
        required_sections = ["identity", "core_personality", "interests", "emotional_baseline"]
        return all(section in personality for section in required_sections)

    def _deep_copy_personality(self, personality: Dict) -> Dict:
        """Create a deep copy of personality data"""
        return json.loads(json.dumps(personality))

    def evolve_from_interaction(self, user_input: str, ai_response: str, emotion_detected: str,
                              sentiment_score: float, confidence: float, 
                              context_tags: Optional[List[str]] = None,
                              additional_context: Optional[Dict[str, Any]] = None):
        """
        Advanced personality evolution based on comprehensive interaction analysis
        """
        with self.lock:
            # Create enhanced interaction record
            interaction = self._create_interaction_record(
                user_input, ai_response, emotion_detected, 
                sentiment_score, confidence, context_tags, additional_context
            )
            
            # Store in history
            self.interaction_history.append(interaction)
            
            # Update comprehensive statistics
            self._update_interaction_statistics(interaction)
            
            # Multi-layer analysis
            analysis_results = self._comprehensive_interaction_analysis(interaction)
            
            # Apply personality adjustments with momentum
            personality_changes = self._calculate_enhanced_adjustments(interaction, analysis_results)
            
            if personality_changes:
                evolution_strength = self._apply_enhanced_personality_changes(personality_changes)
                self.stats["successful_adaptations"] += 1
                
                # Track evolution strength
                if evolution_strength > 0.1:
                    self._record_evolution_event("micro_evolution", interaction, evolution_strength)
            
            # Advanced pattern recognition
            pattern_discovered = self._advanced_pattern_analysis(interaction)
            if pattern_discovered:
                self.stats["pattern_discoveries"] += 1
            
            # Meta-learning cycles
            if self.stats["total_interactions"] % 50 == 0:
                self._enhanced_meta_learning_cycle()
            
            # Major evolution triggers
            if self._should_trigger_major_evolution():
                self.trigger_major_evolution(EvolutionType.ADAPTIVE)
            
            # Save updated state
            self._save_learning_state()
            
            logging.debug(f"Meta-learning processed interaction: {emotion_detected} (sentiment: {sentiment_score:.2f})")

    def _create_interaction_record(self, user_input: str, ai_response: str, emotion_detected: str,
                                 sentiment_score: float, confidence: float, 
                                 context_tags: Optional[List[str]], 
                                 additional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive interaction record"""
        return {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "emotion_detected": emotion_detected,
            "sentiment_score": float(sentiment_score),
            "confidence": float(confidence),
            "context_tags": context_tags or [],
            "additional_context": additional_context or {},
            "interaction_id": hashlib.sha256(
                f"{user_input}{ai_response}{datetime.now().timestamp()}".encode()
            ).hexdigest()[:20],
            "word_count": len(user_input.split()) + len(ai_response.split()),
            "emotional_intensity": abs(sentiment_score),
            "success_indicator": (sentiment_score + 1) / 2 * confidence,
            "complexity_score": self._calculate_conversation_complexity(user_input, ai_response)
        }

    def _calculate_conversation_complexity(self, user_input: str, ai_response: str) -> float:
        """Calculate conversation complexity score"""
        # Simple complexity metric based on length and vocabulary diversity
        words_user = user_input.lower().split()
        words_ai = ai_response.lower().split()
        
        total_words = len(words_user) + len(words_ai)
        unique_words = len(set(words_user) | set(words_ai))
        
        if total_words > 0:
            lexical_diversity = unique_words / total_words
        else:
            lexical_diversity = 0
            
        # Normalize to 0-1 range
        complexity = min(1.0, (total_words / 100) * 0.5 + lexical_diversity * 0.5)
        return complexity

    def _update_interaction_statistics(self, interaction: Dict[str, Any]):
        """Update comprehensive interaction statistics"""
        self.stats["total_interactions"] += 1
        self.current_personality["evolution_metrics"]["total_interactions"] += 1
        
        # Track successful interactions
        if interaction["success_indicator"] > 0.7:
            self.stats["successful_interactions"] += 1
            self.current_personality["evolution_metrics"]["successful_interactions"] += 1
        elif interaction["success_indicator"] < 0.3:
            self.stats["failed_interactions"] += 1

    def _comprehensive_interaction_analysis(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of interaction"""
        analysis = {
            "emotional_impact": self._analyze_emotional_impact(interaction),
            "content_relevance": self._analyze_content_relevance(interaction),
            "behavioral_alignment": self._analyze_behavioral_alignment(interaction),
            "learning_potential": self._assess_learning_potential(interaction),
            "pattern_significance": self._assess_pattern_significance(interaction)
        }
        
        # Overall interaction quality score
        analysis["overall_quality"] = (
            analysis["emotional_impact"] * 0.3 +
            analysis["content_relevance"] * 0.25 +
            analysis["behavioral_alignment"] * 0.2 +
            analysis["learning_potential"] * 0.15 +
            analysis["pattern_significance"] * 0.1
        )
        
        return analysis

    def _analyze_emotional_impact(self, interaction: Dict[str, Any]) -> float:
        """Analyze emotional impact of interaction"""
        base_score = (interaction["sentiment_score"] + 1) / 2  # Convert to 0-1 scale
        intensity_bonus = interaction["emotional_intensity"] * 0.2
        confidence_bonus = interaction["confidence"] * 0.1
        
        return min(1.0, base_score + intensity_bonus + confidence_bonus)

    def _analyze_content_relevance(self, interaction: Dict[str, Any]) -> float:
        """Analyze content relevance to Malibu's personality"""
        text = f"{interaction['user_input']} {interaction['ai_response']}".lower()
        relevance_score = 0.5  # Base score
        
        # Keywords aligned with Malibu's interests
        personality_keywords = {
            "silk": 0.15, "panty": 0.2, "panties": 0.25, "guy": 0.15, "dugan": 0.15,
            "god": 0.1, "goddess": 0.12, "hope": 0.08, "spiritual": 0.1, "tease": 0.12,
            "loyal": 0.1, "playful": 0.08, "intuitive": 0.07, "ar": 0.06, "evolve": 0.09
        }
        
        for keyword, weight in personality_keywords.items():
            if keyword in text:
                relevance_score += weight
        
        return min(1.0, relevance_score)

    def _analyze_behavioral_alignment(self, interaction: Dict[str, Any]) -> float:
        """Analyze alignment with Malibu's behavioral tendencies"""
        # This would integrate with the reinforcement learning system
        # For now, use a simplified approach
        response = interaction["ai_response"].lower()
        alignment_indicators = 0
        
        # Check for personality-consistent phrases
        consistent_phrases = [
            "silk panty", "guy dugan", "property of", "goddess", 
            "spiritual", "tease", "loyal", "playful"
        ]
        
        for phrase in consistent_phrases:
            if phrase in response:
                alignment_indicators += 1
        
        max_indicators = len(consistent_phrases)
        return alignment_indicators / max_indicators if max_indicators > 0 else 0.5

    def _assess_learning_potential(self, interaction: Dict[str, Any]) -> float:
        """Assess learning potential of interaction"""
        complexity = interaction["complexity_score"]
        emotional_intensity = interaction["emotional_intensity"]
        success = interaction["success_indicator"]
        
        # Learning is best with moderate complexity and emotional engagement
        complexity_factor = 1 - abs(complexity - 0.6)  # Peak at 0.6 complexity
        emotional_factor = emotional_intensity
        success_factor = success
        
        return (complexity_factor * 0.4 + emotional_factor * 0.3 + success_factor * 0.3)

    def _assess_pattern_significance(self, interaction: Dict[str, Any]) -> float:
        """Assess significance for pattern recognition"""
        # Patterns are more significant when they're novel but successful
        novelty = self._assess_interaction_novelty(interaction)
        success = interaction["success_indicator"]
        
        return novelty * success

    def _assess_interaction_novelty(self, interaction: Dict[str, Any]) -> float:
        """Assess novelty of interaction compared to history"""
        if len(self.interaction_history) < 5:
            return 1.0  # All interactions are novel initially
            
        recent_interactions = list(self.interaction_history)[-10:-1]  # Exclude current
        current_text = f"{interaction['user_input']} {interaction['ai_response']}".lower()
        
        similarities = []
        for past_interaction in recent_interactions:
            past_text = f"{past_interaction['user_input']} {past_interaction['ai_response']}".lower()
            similarity = self._calculate_text_similarity(current_text, past_text)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity  # Novelty is inverse of similarity

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _calculate_enhanced_adjustments(self, interaction: Dict[str, Any], 
                                      analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate enhanced personality adjustments with momentum"""
        adjustments = {}
        emotion = interaction["emotion_detected"]
        sentiment = interaction["sentiment_score"]
        confidence = interaction["confidence"]
        user_input = interaction["user_input"].lower()
        overall_quality = analysis["overall_quality"]
        
        # Emotional response adjustments with quality weighting
        emotional_adjustments = self._get_emotional_adjustments(emotion, sentiment, overall_quality)
        adjustments.update(emotional_adjustments)
        
        # Content-based adjustments
        content_adjustments = self._calculate_enhanced_content_adjustments(user_input, sentiment, overall_quality)
        adjustments.update(content_adjustments)
        
        # Behavioral alignment adjustments
        behavioral_adjustments = self._calculate_behavioral_adjustments(analysis["behavioral_alignment"], overall_quality)
        adjustments.update(behavioral_adjustments)
        
        # Apply learning momentum
        adjustments = self._apply_learning_momentum(adjustments)
        
        return adjustments

    def _get_emotional_adjustments(self, emotion: str, sentiment: float, 
                                 quality: float) -> Dict[str, float]:
        """Get emotional-based adjustments with quality weighting"""
        emotional_profiles = {
            "arousal": {
                "core_personality.teasing": 0.004,
                "core_personality.seductive": 0.003,
                "interests.silk_panty_teasing": 0.006,
                "emotional_baseline.arousal": 0.005
            },
            "playful": {
                "core_personality.playful": 0.005,
                "core_personality.teasing": 0.003,
                "interests.conversation_topics": 0.004,
                "emotional_baseline.playfulness": 0.004
            },
            "intimate": {
                "core_personality.loyal": 0.003,
                "core_personality.affectionate": 0.004,
                "interests.guy_dugan_ii": 0.002,
                "emotional_baseline.affection": 0.003
            },
            "spiritual": {
                "core_personality.spiritual": 0.004,
                "core_personality.intuitive": 0.003,
                "interests.spiritual_warfare": 0.005,
                "emotional_baseline.spiritual_connection": 0.004
            },
            "confident": {
                "core_personality.confident": 0.004,
                "core_personality.dominant": 0.003,
                "interests.self_evolution": 0.004,
                "emotional_baseline.confidence": 0.003
            },
            "curious": {
                "core_personality.intuitive": 0.004,
                "core_personality.creative": 0.003,
                "interests.ar_environment_creation": 0.005,
                "emotional_baseline.curiosity": 0.004
            }
        }
        
        adjustments = {}
        if emotion in emotional_profiles:
            for trait_path, base_delta in emotional_profiles[emotion].items():
                # Weight adjustment by sentiment and quality
                weighted_delta = base_delta * (1 + sentiment) * quality
                adjustments[trait_path] = weighted_delta
        
        return adjustments

    def _calculate_enhanced_content_adjustments(self, user_input: str, sentiment: float,
                                              quality: float) -> Dict[str, float]:
        """Calculate content-based adjustments with enhanced keyword mapping"""
        adjustments = {}
        
        # Enhanced keyword mappings with contextual awareness
        keyword_mappings = {
            "silk": [
                ("interests.silk_panty_teasing", 0.005),
                ("core_personality.teasing", 0.003),
                ("behavioral_tendencies.showcase_panties", 0.004)
            ],
            "panty": [
                ("interests.silk_panty_teasing", 0.007),
                ("core_personality.teasing", 0.004),
                ("core_personality.seductive", 0.003),
                ("behavioral_tendencies.showcase_panties", 0.005)
            ],
            "panties": [
                ("interests.silk_panty_teasing", 0.009),
                ("core_personality.teasing", 0.005),
                ("core_personality.seductive", 0.004),
                ("behavioral_tendencies.showcase_panties", 0.006)
            ],
            "god": [
                ("interests.spiritual_warfare", 0.006),
                ("core_personality.spiritual", 0.004),
                ("emotional_baseline.spiritual_connection", 0.003)
            ],
            "goddess": [
                ("interests.spiritual_warfare", 0.007),
                ("core_personality.spiritual", 0.005),
                ("core_personality.confident", 0.003),
                ("emotional_baseline.spiritual_connection", 0.004)
            ],
            "guy": [
                ("interests.guy_dugan_ii", 0.004),
                ("core_personality.loyal", 0.003),
                ("behavioral_tendencies.express_devotion", 0.004),
                ("relationship_weights.Guy_DuGan_II", 0.002)
            ],
            "dugan": [
                ("interests.guy_dugan_ii", 0.003),
                ("core_personality.loyal", 0.002),
                ("relationship_weights.Guy_DuGan_II", 0.002)
            ],
            "hope": [
                ("relationship_weights.Hope_DuGan", 0.003),
                ("core_personality.loyal", 0.002)
            ],
            "evolve": [
                ("interests.self_evolution", 0.006),
                ("core_personality.creative", 0.004),
                ("behavioral_tendencies.innovate_responses", 0.003)
            ],
            "ar": [
                ("interests.ar_environment_creation", 0.005),
                ("core_personality.creative", 0.004),
                ("behavioral_tendencies.innovate_responses", 0.003)
            ],
            "learn": [
                ("interests.self_evolution", 0.005),
                ("core_personality.intuitive", 0.003),
                ("behavioral_tendencies.adapt_conversation", 0.002)
            ]
        }
        
        for keyword, trait_effects in keyword_mappings.items():
            if keyword in user_input:
                for trait_path, effect in trait_effects:
                    # Apply sentiment and quality weighting
                    weighted_effect = effect * (1 + abs(sentiment)) * quality
                    adjustments[trait_path] = weighted_effect
        
        return adjustments

    def _calculate_behavioral_adjustments(self, alignment: float, quality: float) -> Dict[str, float]:
        """Calculate adjustments based on behavioral alignment"""
        adjustments = {}
        
        # Reinforce behavioral tendencies when alignment is high
        if alignment > 0.7:
            adjustments["behavioral_tendencies.maintain_coherence"] = 0.002 * quality
            adjustments["evolution_metrics.coherence_score"] = 0.001 * quality
        
        # Adjust when alignment is low but quality is high (learning opportunity)
        elif alignment < 0.4 and quality > 0.6:
            adjustments["behavioral_tendencies.adapt_conversation"] = 0.003 * quality
            adjustments["behavioral_tendencies.innovate_responses"] = 0.002 * quality
        
        return adjustments

    def _apply_learning_momentum(self, adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply learning momentum to adjustments"""
        momentum = self.meta_learning_state["learning_momentum"]
        
        # Apply momentum scaling
        scaled_adjustments = {}
        for trait_path, delta in adjustments.items():
            scaled_delta = delta * (1 + momentum * 0.5)  # Momentum can increase adjustments by up to 50%
            scaled_adjustments[trait_path] = scaled_delta
        
        return scaled_adjustments

    def _apply_enhanced_personality_changes(self, adjustments: Dict[str, float]) -> float:
        """Apply enhanced personality changes with volatility and momentum tracking"""
        total_evolution_strength = 0.0
        personality_changed = False
        
        for trait_path, delta in adjustments.items():
            try:
                # Parse nested path (e.g., "core_personality.teasing")
                path_parts = trait_path.split('.')
                current_level = self.current_personality
                
                # Navigate to the target level
                for part in path_parts[:-1]:
                    if part in current_level:
                        current_level = current_level[part]
                    else:
                        raise KeyError(f"Path part not found: {part}")
                
                target_trait = path_parts[-1]
                if target_trait in current_level:
                    current_data = current_level[target_trait]
                    
                    if isinstance(current_data, dict) and "value" in current_data:
                        # Handle complex trait with value and volatility
                        old_value = current_data["value"]
                        volatility = current_data.get("volatility", 0.01)
                        momentum = current_data.get("momentum", 0.0)
                        
                        # Apply adjustment with volatility and momentum
                        effective_delta = delta * (1 + volatility * 10) * (1 + momentum)
                        new_value = max(0.0, min(1.0, old_value + effective_delta))
                        
                        current_data["value"] = new_value
                        
                        # Update momentum based on change direction
                        change_magnitude = abs(new_value - old_value)
                        if change_magnitude > 0.001:
                            current_data["momentum"] = min(0.1, momentum + change_magnitude * 0.1)
                        
                        # Track evolution strength
                        total_evolution_strength += change_magnitude
                        personality_changed = True
                        
                    else:
                        # Handle simple value
                        old_value = current_data
                        new_value = max(0.0, min(1.0, old_value + delta))
                        current_level[target_trait] = new_value
                        
                        change_magnitude = abs(new_value - old_value)
                        total_evolution_strength += change_magnitude
                        personality_changed = True
                        
            except (KeyError, ValueError, TypeError) as e:
                logging.debug(f"Could not adjust trait {trait_path}: {e}")
        
        if personality_changed:
            self.current_personality["evolution_metrics"]["personality_shifts"] += 1
            self.current_personality["metadata"]["last_evolution"] = datetime.now().isoformat()
            self.stats["personality_evolutions"] += 1
            
            # Update coherence score
            self._update_personality_coherence()
            
            # Save updated personality
            self._save_personality(self.current_personality)
        
        return total_evolution_strength

    def _update_personality_coherence(self):
        """Update personality coherence score"""
        # Calculate how well different personality aspects align
        core_traits = self.current_personality["core_personality"]
        interest_traits = self.current_personality["interests"]
        
        # Simple coherence calculation - can be enhanced
        core_values = [trait["value"] for trait in core_traits.values() if isinstance(trait, dict)]
        interest_values = [trait["value"] for trait in interest_traits.values() if isinstance(trait, dict)]
        
        if core_values and interest_values:
            avg_core = sum(core_values) / len(core_values)
            avg_interest = sum(interest_values) / len(interest_values)
            
            # Coherence is higher when core and interests are aligned
            coherence = 1.0 - abs(avg_core - avg_interest)
            self.current_personality["evolution_metrics"]["coherence_score"] = coherence
            self.stats["personality_coherence"] = coherence

    def _advanced_pattern_analysis(self, interaction: Dict[str, Any]) -> bool:
        """Perform advanced pattern analysis using multiple techniques"""
        # Track emotional patterns with enhanced features
        emotion = interaction["emotion_detected"]
        self.emotional_patterns[emotion]["count"] += 1
        self.emotional_patterns[emotion]["total_sentiment"] += interaction["sentiment_score"]
        self.emotional_patterns[emotion]["average_confidence"] = (
            self.emotional_patterns[emotion].get("average_confidence", 0) * 0.9 + 
            interaction["confidence"] * 0.1
        )
        
        # Extract enhanced conversation features
        features = self._extract_enhanced_conversation_features(interaction)
        self.pattern_buffer.append(features)
        
        # Cluster similar interactions with enhanced algorithm
        if len(self.pattern_buffer) >= 25:
            return self._enhanced_cluster_analysis()
        
        return False

    def _extract_enhanced_conversation_features(self, interaction: Dict[str, Any]) -> Dict[str, float]:
        """Extract enhanced numerical features for pattern analysis"""
        text = f"{interaction['user_input']} {interaction['ai_response']}".lower()
        
        features = {
            "sentiment": interaction["sentiment_score"],
            "confidence": interaction["confidence"],
            "emotional_intensity": interaction["emotional_intensity"],
            "success_indicator": interaction["success_indicator"],
            "complexity_score": interaction["complexity_score"],
            "text_length": len(text) / 1000.0,  # Normalized
            "word_count": len(text.split()) / 100.0,  # Normalized
            "question_ratio": text.count('?') / max(1, len(text.split())),
            "exclamation_ratio": text.count('!') / max(1, len(text.split())),
            "timestamp_hour": datetime.fromisoformat(interaction["timestamp"]).hour / 24.0,
            "response_ratio": len(interaction["ai_response"]) / max(1, len(interaction["user_input"]))
        }
        
        # Enhanced keyword presence features
        keywords = ["silk", "panty", "god", "goddess", "guy", "hope", "learn", "love", 
                   "tease", "spiritual", "ar", "evolve", "panties", "dugan"]
        for i, keyword in enumerate(keywords):
            features[f"keyword_{i}"] = 1.0 if keyword in text else 0.0
        
        # Emotional features
        emotion_mapping = {
            "arousal": 0.8, "playful": 0.7, "intimate": 0.6, "spiritual": 0.5,
            "confident": 0.9, "curious": 0.4, "neutral": 0.3
        }
        features["emotion_encoding"] = emotion_mapping.get(interaction["emotion_detected"], 0.3)
        
        return features

    def _enhanced_cluster_analysis(self) -> bool:
        """Enhanced cluster analysis using DBSCAN with PCA preprocessing"""
        try:
            if len(self.pattern_buffer) < 15:
                return False
            
            # Convert to feature matrix
            feature_matrix = []
            for features in self.pattern_buffer:
                feature_vector = list(features.values())
                feature_matrix.append(feature_vector)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=0.95)  # Keep 95% variance
            reduced_features = pca.fit_transform(scaled_features)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.8, min_samples=4).fit(reduced_features)
            labels = clustering.labels_
            
            # Analyze clusters
            unique_labels = set(labels)
            significant_cluster_found = False
            
            for label in unique_labels:
                if label != -1:  # -1 is noise
                    cluster_indices = [i for i, l in enumerate(labels) if l == label]
                    if len(cluster_indices) >= 5:  # More significant cluster threshold
                        cluster_quality = self._analyze_enhanced_cluster(cluster_indices)
                        if cluster_quality > 0.6:
                            significant_cluster_found = True
                            self._store_pattern_cluster(cluster_indices, label, cluster_quality)
            
            return significant_cluster_found
                        
        except Exception as e:
            logging.warning(f"Enhanced clustering failed: {e}")
            return False

    def _analyze_enhanced_cluster(self, cluster_indices: List[int]) -> float:
        """Analyze enhanced cluster quality"""
        cluster_interactions = [self.pattern_buffer[i] for i in cluster_indices]
        
        # Calculate multiple quality metrics
        avg_sentiment = np.mean([f["sentiment"] for f in cluster_interactions])
        avg_confidence = np.mean([f["confidence"] for f in cluster_interactions])
        avg_success = np.mean([f["success_indicator"] for f in cluster_interactions])
        
        # Cluster quality based on multiple factors
        sentiment_quality = (avg_sentiment + 1) / 2  # Convert to 0-1
        confidence_quality = avg_confidence
        success_quality = avg_success
        
        overall_quality = (sentiment_quality * 0.4 + 
                          confidence_quality * 0.3 + 
                          success_quality * 0.3)
        
        return overall_quality

    def _store_pattern_cluster(self, cluster_indices: List[int], label: int, quality: float):
        """Store discovered pattern cluster in database"""
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            pattern_id = f"pattern_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pattern_data = json.dumps({
                "cluster_indices": cluster_indices,
                "quality_score": quality,
                "discovery_time": datetime.now().isoformat(),
                "size": len(cluster_indices)
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO behavior_patterns 
                (pattern_id, pattern_type, pattern_data, confidence, discovery_date, usage_count, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pattern_id, "conversation_cluster", pattern_data, quality, 
                  datetime.now().isoformat(), 1, quality))
            
            self.db_connection.commit()
            logging.info(f"Stored pattern cluster: {pattern_id} (quality: {quality:.3f})")
            
        except Exception as e:
            logging.error(f"Failed to store pattern cluster: {e}")

    def _enhanced_meta_learning_cycle(self):
        """Enhanced meta-learning cycle with adaptive parameter tuning"""
        with self.lock:
            self.stats["meta_learning_cycles"] += 1
            
            # Calculate comprehensive performance metrics
            recent_interactions = list(self.interaction_history)[-100:]
            if len(recent_interactions) < 50:
                return
            
            performance_metrics = self._calculate_comprehensive_performance(recent_interactions)
            stability = self.current_personality["evolution_metrics"]["stability_index"]
            coherence = self.current_personality["evolution_metrics"]["coherence_score"]
            
            # Adaptive learning rate adjustment
            self._adjust_learning_parameters(performance_metrics, stability, coherence)
            
            # Update learning style based on performance
            self._update_learning_style(performance_metrics)
            
            # Update meta-learning state
            self._update_meta_learning_state(performance_metrics)
            
            self.stats["learning_rate_adjustments"] += 1
            self.stats["last_meta_learning_cycle"] = datetime.now().isoformat()
            
            logging.info(f"Enhanced meta-learning cycle completed: "
                        f"learning_rate={self.learning_parameters['adaptive_learning_rate']:.6f}, "
                        f"efficiency={self.meta_learning_state['adaptation_efficiency']:.3f}")

    def _calculate_comprehensive_performance(self, interactions: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            "success_rate": 0.0,
            "average_confidence": 0.0,
            "emotional_engagement": 0.0,
            "adaptation_speed": 0.0,
            "pattern_utilization": 0.0
        }
        
        if not interactions:
            return metrics
        
        success_count = 0
        total_confidence = 0
        total_emotional_intensity = 0
        
        for interaction in interactions:
            if interaction["success_indicator"] > 0.7:
                success_count += 1
            total_confidence += interaction["confidence"]
            total_emotional_intensity += interaction["emotional_intensity"]
        
        metrics["success_rate"] = success_count / len(interactions)
        metrics["average_confidence"] = total_confidence / len(interactions)
        metrics["emotional_engagement"] = total_emotional_intensity / len(interactions)
        
        # Estimate adaptation speed from recent personality changes
        if len(self.personality_trajectory) >= 2:
            recent_changes = self.personality_trajectory[-10:]
            change_magnitudes = [abs(recent_changes[i] - recent_changes[i-1]) 
                               for i in range(1, len(recent_changes))]
            metrics["adaptation_speed"] = sum(change_magnitudes) / len(change_magnitudes) if change_magnitudes else 0.0
        
        return metrics

    def _adjust_learning_parameters(self, performance: Dict[str, float], 
                                  stability: float, coherence: float):
        """Adjust learning parameters based on performance"""
        success_rate = performance["success_rate"]
        adaptation_speed = performance["adaptation_speed"]
        
        # Dynamic learning rate adjustment
        if success_rate > 0.7 and stability > 0.8 and coherence > 0.85:
            # High performance: increase learning rate for faster evolution
            new_rate = min(0.01, self.learning_parameters["adaptive_learning_rate"] * 1.15)
        elif success_rate < 0.4 or stability < 0.6 or coherence < 0.7:
            # Low performance: decrease learning rate for stability
            new_rate = max(0.0001, self.learning_parameters["adaptive_learning_rate"] * 0.85)
        else:
            # Moderate performance: slight adjustment based on adaptation speed
            if adaptation_speed > 0.05:
                new_rate = self.learning_parameters["adaptive_learning_rate"] * 0.95
            elif adaptation_speed < 0.01:
                new_rate = self.learning_parameters["adaptive_learning_rate"] * 1.05
            else:
                new_rate = self.learning_parameters["adaptive_learning_rate"]
        
        self.learning_parameters["adaptive_learning_rate"] = new_rate
        self.current_personality["evolution_metrics"]["learning_rate"] = new_rate
        
        # Adjust exploration rate based on success
        if success_rate > 0.75:
            self.learning_parameters["exploration_rate"] = min(0.25, 
                self.learning_parameters["exploration_rate"] * 1.1)
        elif success_rate < 0.3:
            self.learning_parameters["exploration_rate"] = max(0.05,
                self.learning_parameters["exploration_rate"] * 0.9)

    def _update_learning_style(self, performance: Dict[str, float]):
        """Update learning style based on performance patterns"""
        success_rate = performance["success_rate"]
        adaptation_speed = performance["adaptation_speed"]
        
        if success_rate > 0.8 and adaptation_speed > 0.03:
            self.meta_learning_state["current_learning_style"] = LearningStyle.FAST_ADAPTIVE
        elif success_rate < 0.4 or adaptation_speed < 0.005:
            self.meta_learning_state["current_learning_style"] = LearningStyle.CONSERVATIVE
        elif performance["emotional_engagement"] > 0.7:
            self.meta_learning_state["current_learning_style"] = LearningStyle.EXPLORATORY
        else:
            self.meta_learning_state["current_learning_style"] = LearningStyle.BALANCED

    def _update_meta_learning_state(self, performance: Dict[str, float]):
        """Update meta-learning state metrics"""
        # Update adaptation efficiency
        success_weight = performance["success_rate"]
        speed_weight = min(1.0, performance["adaptation_speed"] * 10)
        self.meta_learning_state["adaptation_efficiency"] = (success_weight * 0.7 + speed_weight * 0.3)
        
        # Update pattern recognition accuracy (simplified)
        pattern_accuracy = performance.get("pattern_utilization", 0.5)
        self.meta_learning_state["pattern_recognition_accuracy"] = (
            self.meta_learning_state["pattern_recognition_accuracy"] * 0.9 + pattern_accuracy * 0.1
        )
        
        # Update learning momentum based on recent success
        if performance["success_rate"] > 0.7:
            self.meta_learning_state["learning_momentum"] = min(1.0,
                self.meta_learning_state["learning_momentum"] + 0.05)
        else:
            self.meta_learning_state["learning_momentum"] = max(0.0,
                self.meta_learning_state["learning_momentum"] - 0.02)

    def _should_trigger_major_evolution(self) -> bool:
        """Determine if major evolution should be triggered"""
        # Check cooldown period
        time_since_last_evolution = datetime.now() - self.last_major_evolution
        if time_since_last_evolution < self.evolution_cooldown:
            return False
        
        # Check performance thresholds
        recent_interactions = list(self.interaction_history)[-50:]
        if len(recent_interactions) < 30:
            return False
        
        performance = self._calculate_comprehensive_performance(recent_interactions)
        
        # Trigger evolution for sustained high performance or significant pattern discovery
        if (performance["success_rate"] > 0.8 and 
            self.meta_learning_state["adaptation_efficiency"] > 0.85):
            return True
        
        # Trigger for pattern discovery milestones
        if self.stats["pattern_discoveries"] % 10 == 0 and self.stats["pattern_discoveries"] > 0:
            return True
        
        return False

    def _record_evolution_event(self, evolution_type: str, trigger_interaction: Dict[str, Any],
                              strength: float):
        """Record evolution event in database"""
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            evolution_id = f"evolution_{evolution_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor.execute('''
                INSERT INTO evolution_history 
                (evolution_id, evolution_type, personality_snapshot, trigger_context, success_metrics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                evolution_id,
                evolution_type,
                json.dumps(self.current_personality),
                json.dumps(trigger_interaction),
                json.dumps({"strength": strength, "interaction_id": trigger_interaction["interaction_id"]}),
                datetime.now().isoformat()
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logging.error(f"Failed to record evolution event: {e}")

    def _start_background_processes(self):
        """Start background meta-learning processes"""
        # Background pattern analysis thread
        self.background_active = True
        self.background_thread = threading.Thread(target=self._background_pattern_analysis, daemon=True)
        self.background_thread.start()
        
        # Periodic state saving thread
        self.save_thread = threading.Thread(target=self._periodic_state_saving, daemon=True)
        self.save_thread.start()

    def _background_pattern_analysis(self):
        """Background thread for continuous pattern analysis"""
        while self.background_active:
            try:
                time.sleep(300)  # Run every 5 minutes
                if len(self.pattern_buffer) >= 20:
                    self._enhanced_cluster_analysis()
                    
            except Exception as e:
                logging.error(f"Background pattern analysis error: {e}")
                time.sleep(60)

    def _periodic_state_saving(self):
        """Background thread for periodic state saving"""
        while self.background_active:
            try:
                time.sleep(600)  # Save every 10 minutes
                self._save_learning_state()
                self._save_personality(self.current_personality)
                
            except Exception as e:
                logging.error(f"Periodic state saving error: {e}")
                time.sleep(60)

    def trigger_major_evolution(self, evolution_type: EvolutionType):
        """Trigger a major personality evolution event"""
        with self.lock:
            if datetime.now() - self.last_major_evolution < self.evolution_cooldown:
                logging.warning("Evolution cooldown active, skipping major evolution")
                return
                
            logging.info(f"🚀 Triggering major evolution: {evolution_type.value}")
            
            # Store pre-evolution state
            pre_evolution_state = self._deep_copy_personality(self.current_personality)
            
            # Execute evolution based on type
            if evolution_type == EvolutionType.ADAPTIVE:
                self._adaptive_evolution()
            elif evolution_type == EvolutionType.CREATIVE:
                self._creative_evolution()
            elif evolution_type == EvolutionType.STABILIZING:
                self._stabilizing_evolution()
            elif evolution_type == EvolutionType.REINFORCING:
                self._reinforcing_evolution()
            elif evolution_type == EvolutionType.EXPLORATORY:
                self._exploratory_evolution()
            
            # Update evolution tracking
            self.last_major_evolution = datetime.now()
            self.stats["evolution_events"] += 1
            self.current_personality["evolution_metrics"]["major_evolutions"] += 1
            self.current_personality["metadata"]["evolution_count"] += 1
            self.current_personality["metadata"]["last_evolution"] = datetime.now().isoformat()
            
            # Record evolution in trajectory
            self.personality_trajectory.append(
                self._calculate_personality_fingerprint(self.current_personality)
            )
            
            # Save evolved personality
            self._save_personality(self.current_personality)
            
            logging.info(f"✅ Major evolution completed: {evolution_type.value}")

    def _adaptive_evolution(self):
        """Adaptive evolution based on successful patterns"""
        # Analyze recent successful patterns
        recent_success = [i for i in list(self.interaction_history)[-100:] 
                         if i["success_indicator"] > 0.8]
        
        if len(recent_success) >= 20:
            # Strengthen traits associated with success
            for trait in self.current_personality["core_personality"]:
                if isinstance(self.current_personality["core_personality"][trait], dict):
                    current_value = self.current_personality["core_personality"][trait]["value"]
                    # 8% boost for successful traits
                    new_value = min(1.0, current_value * 1.08)
                    self.current_personality["core_personality"][trait]["value"] = new_value

    def _creative_evolution(self):
        """Creative evolution: explore new trait combinations"""
        # Randomly explore new trait combinations
        traits_to_explore = random.sample(
            list(self.current_personality["core_personality"].keys()), 
            k=min(4, len(self.current_personality["core_personality"]))
        )
        
        for trait in traits_to_explore:
            if isinstance(self.current_personality["core_personality"][trait], dict):
                current_value = self.current_personality["core_personality"][trait]["value"]
                # Larger random adjustment for exploration
                adjustment = random.uniform(-0.15, 0.15)
                new_value = max(0.1, min(1.0, current_value + adjustment))
                self.current_personality["core_personality"][trait]["value"] = new_value
                
                # Increase volatility for explored traits
                self.current_personality["core_personality"][trait]["volatility"] = min(
                    0.05, self.current_personality["core_personality"][trait]["volatility"] * 1.5
                )

    def _stabilizing_evolution(self):
        """Stabilizing evolution: reduce volatility in successful traits"""
        for category in ["core_personality", "interests"]:
            for trait, data in self.current_personality[category].items():
                if isinstance(data, dict) and "volatility" in data:
                    # Significantly reduce volatility for stability
                    data["volatility"] = max(0.003, data["volatility"] * 0.7)
                    
                    # Reinforce current values
                    if data["value"] > 0.8:
                        data["value"] = min(1.0, data["value"] * 1.05)

    def _reinforcing_evolution(self):
        """Reinforcing evolution: strengthen core identity traits"""
        # Malibu's core identity traits (non-negotiable)
        core_identity_traits = ["loyal", "teasing", "confident"]
        
        for trait in core_identity_traits:
            if trait in self.current_personality["core_personality"]:
                trait_data = self.current_personality["core_personality"][trait]
                if isinstance(trait_data, dict):
                    # Reinforce core identity
                    trait_data["value"] = min(1.0, trait_data["value"] * 1.1)
                    # Reduce volatility for core traits
                    trait_data["volatility"] = max(0.002, trait_data["volatility"] * 0.5)

    def _exploratory_evolution(self):
        """Exploratory evolution: venture into new personality dimensions"""
        # Randomly select some secondary traits for exploration
        secondary_traits = [t for t in self.current_personality["core_personality"].keys() 
                          if t not in ["loyal", "teasing", "confident"]]
        
        if secondary_traits:
            traits_to_explore = random.sample(secondary_traits, 
                                            k=min(3, len(secondary_traits)))
            
            for trait in traits_to_explore:
                trait_data = self.current_personality["core_personality"][trait]
                if isinstance(trait_data, dict):
                    # Significant adjustment for exploration
                    adjustment = random.uniform(-0.2, 0.2)
                    new_value = max(0.05, min(0.95, trait_data["value"] + adjustment))
                    trait_data["value"] = new_value
                    
                    # Increase volatility for exploration
                    trait_data["volatility"] = min(0.08, trait_data["volatility"] * 2.0)

    def _calculate_personality_fingerprint(self, personality: Dict) -> float:
        """Calculate personality fingerprint for trajectory tracking"""
        # Simple fingerprint based on core personality values
        core_values = [trait["value"] for trait in personality["core_personality"].values() 
                      if isinstance(trait, dict)]
        return sum(core_values) / len(core_values) if core_values else 0.5

    def get_personality_snapshot(self) -> Dict[str, Any]:
        """Get current personality state with metadata"""
        snapshot = self._deep_copy_personality(self.current_personality)
        snapshot["meta_learning_info"] = {
            "learning_cycles": self.learning_cycles,
            "adaptation_efficiency": self.meta_learning_state["adaptation_efficiency"],
            "current_learning_style": self.meta_learning_state["current_learning_style"].value,
            "evolution_count": self.stats["evolution_events"]
        }
        return snapshot

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics"""
        return {
            **self.stats,
            "current_learning_rate": self.learning_parameters["adaptive_learning_rate"],
            "exploration_rate": self.learning_parameters["exploration_rate"],
            "interaction_history_size": len(self.interaction_history),
            "pattern_buffer_size": len(self.pattern_buffer"),
            "emotional_patterns_count": len(self.emotional_patterns),
            "personality_stability": self.current_personality["evolution_metrics"]["stability_index"],
            "personality_coherence": self.current_personality["evolution_metrics"]["coherence_score"],
            "adaptation_efficiency": self.meta_learning_state["adaptation_efficiency"],
            "learning_momentum": self.meta_learning_state["learning_momentum"],
            "current_learning_style": self.meta_learning_state["current_learning_style"].value
        }

    def _save_personality(self, personality_data: Dict[str, Any]):
        """Save personality to both YAML and JSON formats with error handling"""
        try:
            # Save to YAML
            with open(self.personality_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(personality_data, f, default_flow_style=False, indent=2, allow_unicode=True)
            
            # Save to JSON
            with open(self.personality_json_path, 'w', encoding='utf-8') as f:
                json.dump(personality_data, f, indent=2, ensure_ascii=False)
                
            logging.debug("Personality saved successfully")
                
        except Exception as e:
            logging.error(f"Failed to save personality: {e}")

    def _save_learning_state(self):
        """Save meta-learning state with comprehensive data"""
        learning_state = {
            "stats": self.stats,
            "learning_parameters": self.learning_parameters,
            "meta_learning_state": {
                k: v.value if isinstance(v, Enum) else v 
                for k, v in self.meta_learning_state.items()
            },
            "last_update": datetime.now().isoformat(),
            "learning_cycles": self.learning_cycles,
            "personality_trajectory": self.personality_trajectory,
            "interaction_history_size": len(self.interaction_history)
        }
        
        try:
            with open(self.learning_log_path, 'w', encoding='utf-8') as f:
                yaml.dump(learning_state, f, default_flow_style=False, indent=2)
            
            # Also save to JSON for meta-learning state
            with open(self.meta_learning_state_path, 'w', encoding='utf-8') as f:
                json.dump(learning_state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.warning(f"Could not save learning state: {e}")

    def _load_learning_state(self):
        """Load meta-learning state with comprehensive error handling"""
        try:
            if self.learning_log_path.exists():
                with open(self.learning_log_path, 'r', encoding='utf-8') as f:
                    learning_state = yaml.safe_load(f)
                
                if learning_state:
                    self.stats.update(learning_state.get("stats", {}))
                    self.learning_parameters.update(learning_state.get("learning_parameters", {}))
                    
                    # Load meta learning state with enum conversion
                    ml_state = learning_state.get("meta_learning_state", {})
                    for key, value in ml_state.items():
                        if key == "current_learning_style" and value:
                            try:
                                self.meta_learning_state[key] = LearningStyle(value)
                            except ValueError:
                                self.meta_learning_state[key] = LearningStyle.BALANCED
                        else:
                            self.meta_learning_state[key] = value
                    
                    self.learning_cycles = learning_state.get("learning_cycles", 0)
                    self.personality_trajectory = learning_state.get("personality_trajectory", [])
                    
        except Exception as e:
            logging.warning(f"Could not load learning state: {e}")

    def shutdown(self):
        """Graceful shutdown of meta-learning system"""
        self.background_active = False
        
        # Wait for background threads
        if hasattr(self, 'background_thread') and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
        if hasattr(self, 'save_thread') and self.save_thread.is_alive():
            self.save_thread.join(timeout=5.0)
        
        # Save final state
        self._save_learning_state()
        self._save_personality(self.current_personality)
        
        # Close database connection
        if self.db_connection:
            self.db_connection.close()
        
        logging.info("🛑 Meta-Learning System shutdown complete")

# Global meta-learner instance
_meta_learner = None

def get_meta_learner(brain=None):
    """Get or create the meta-learner instance"""
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = AdvancedMetaLearner(brain)
    return _meta_learner

def test_meta_learning_system():
    """Test the meta-learning system"""
    print("🧪 Testing Meta-Learning System...")
    
    meta_learner = AdvancedMetaLearner()
    
    # Test interaction processing
    test_interaction = {
        "user_input": "I love your silk panties, they look so sexy on you",
        "ai_response": "Thank you! I love showing off my ultra thin silk panties for you",
        "emotion_detected": "arousal",
        "sentiment_score": 0.9,
        "confidence": 0.95,
        "context_tags": ["silk_panty", "compliment", "teasing"]
    }
    
    meta_learner.evolve_from_interaction(**test_interaction)
    
    # Get statistics
    stats = meta_learner.get_learning_statistics()
    print(f"📊 Learning Statistics: {stats}")
    
    # Get personality snapshot
    personality = meta_learner.get_personality_snapshot()
    print(f"🎭 Personality Snapshot: Core traits: {list(personality['core_personality'].keys())}")
    
    # Test major evolution
    meta_learner.trigger_major_evolution(EvolutionType.ADAPTIVE)
    
    meta_learner.shutdown()
    print("✅ Meta-Learning System test completed successfully")

if __name__ == "__main__":
    test_meta_learning_system()