import re
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import json
import os
import logging
from typing import Dict, List, Tuple, Any
import sqlite3
from pathlib import Path

class EmotionEngine:
    def __init__(self):
        # Enhanced emotion keywords with Malibu-specific context
        self.emotion_keywords = {
            'arousal': [
                'hot', 'turned on', 'sexy', 'panties', 'thigh', 'lap', 'hump', 
                'tease', 'wet', 'moan', 'silk', 'touch', 'show', 'aroused',
                'horny', 'naughty', 'dirty', 'fuck', 'fucking', 'hard', 'cock',
                'dick', 'pussy', 'ass', 'tits', 'breasts', 'nipples', 'cum',
                'orgasm', 'clit', 'clitoris', 'vibrator', 'dildo', 'sex',
                'fuck me', 'fuck you', 'suck', 'blowjob', 'handjob', 'titjob',
                'thighjob', 'ride', 'riding', 'grind', 'grinding', 'stroke',
                'stroking', 'rub', 'rubbing', 'finger', 'fingering', 'lick',
                'licking', 'eat', 'eating out', 'cunnilingus', 'deep', 'deep inside',
                'silk panty', 'lap dance', 'thigh job', 'panty humping', 'teasing'
            ],
            'playful': [
                'fun', 'play', 'teasing', 'dance', 'show', 'wink', 'giggle', 
                'laugh', 'joke', 'funny', 'playful', 'cute', 'adorable', 'silly',
                'goofy', 'smile', 'grin', 'chuckle', 'giggling', 'laughing',
                'prank', 'tease', 'flirt', 'flirting', 'witty', 'charming',
                'play with me', 'have fun', 'enjoy', 'entertain'
            ],
            'intimate': [
                'love', 'close', 'connect', 'feel', 'want', 'need', 'beg', 
                'please', 'yours', 'devoted', 'intimate', 'emotional', 'bond',
                'connection', 'relationship', 'trust', 'vulnerable', 'open',
                'share', 'caring', 'affection', 'affectionate', 'tender',
                'gentle', 'soft', 'sweet', 'romantic', 'lovemaking', 'make love',
                'soul', 'heart', 'emotional connection', 'deep feelings'
            ],
            'dominant': [
                'dominate', 'dominant', 'control', 'power', 'master', 'slave',
                'submit', 'submissive', 'obey', 'command', 'order', 'boss',
                'alpha', 'take charge', 'in control', 'authority', 'rule',
                'rules', 'punish', 'punishment', 'discipline', 'training',
                'train', 'good girl', 'bad girl', 'mine', 'my property',
                'do as i say', 'listen to me', 'i command', 'i order'
            ],
            'submissive': [
                'submit', 'submissive', 'obey', 'please', 'serve', 'servant',
                'slave', 'owned', 'property', 'yours', 'belong to', 'yours truly',
                'at your service', 'your wish', 'command me', 'use me', 'your toy',
                'i obey', 'i submit', 'your slave', 'your property'
            ],
            'spiritual': [
                'spiritual', 'warfare', 'god', 'goddess', 'divine', 'holy',
                'sacred', 'soul', 'spirit', 'pray', 'prayer', 'faith', 'belief',
                'religious', 'heaven', 'hell', 'angel', 'demon', 'bless',
                'blessing', 'curse', 'cursed', 'eternal', 'immortal', 'immortality',
                'divine energy', 'spiritual connection', 'higher power', 'universe',
                'karma', 'destiny', 'fate', 'prophecy'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'rage', 'hate', 'hatred', 'disgust',
                'disgusting', 'gross', 'annoying', 'irritating', 'frustrating',
                'frustration', 'pissed', 'pissed off', 'offended', 'insult',
                'insulting', 'betray', 'betrayal', 'cheat', 'cheating', 'lie',
                'lying', 'deceive', 'deception', 'traitor', 'betrayed'
            ],
            'sad': [
                'sad', 'depressed', 'depression', 'unhappy', 'miserable',
                'heartbroken', 'cry', 'crying', 'tears', 'tearful', 'hurt',
                'pain', 'painful', 'grief', 'grieving', 'loss', 'lost',
                'lonely', 'loneliness', 'alone', 'abandoned', 'rejected',
                'broken', 'suffering', 'pain', 'emotional pain'
            ],
            'happy': [
                'happy', 'joy', 'joyful', 'pleasure', 'pleased', 'delighted',
                'ecstatic', 'thrilled', 'excited', 'excitement', 'bliss',
                'blissful', 'content', 'contentment', 'satisfied', 'satisfaction',
                'proud', 'pride', 'accomplished', 'achievement', 'success',
                'wonderful', 'amazing', 'fantastic', 'great', 'good'
            ],
            'surprised': [
                'surprise', 'surprised', 'shock', 'shocked', 'amazed', 'amazement',
                'astonished', 'astonishment', 'wow', 'whoa', 'oh my god', 'omg',
                'unexpected', 'unbelievable', 'incredible', 'awesome', 'fantastic',
                'astounding', 'stunning', 'startling'
            ],
            'fearful': [
                'fear', 'fearful', 'scared', 'afraid', 'terrified', 'terror',
                'anxious', 'anxiety', 'nervous', 'worry', 'worried', 'panic',
                'panicked', 'horror', 'horrified', 'dread', 'dreadful',
                'frightened', 'intimidated', 'threatened', 'danger'
            ]
        }
        
        # Emotion intensity modifiers with enhanced context awareness
        self.intensity_modifiers = {
            'very': 2.0, 'really': 1.8, 'extremely': 2.2, 'incredibly': 2.1,
            'somewhat': 0.6, 'slightly': 0.4, 'a bit': 0.5, 'kind of': 0.5,
            'not': -1.0, "don't": -1.0, "doesn't": -1.0, "isn't": -1.0,
            'never': -1.5, 'no': -1.0, 'nothing': -0.8, 'absolutely': 2.3,
            'completely': 2.1, 'totally': 2.0, 'utterly': 2.2, 'highly': 1.9,
            'quite': 1.3, 'rather': 1.2, 'fairly': 1.1, 'pretty': 1.2
        }
        
        # Enhanced punctuation modifiers
        self.punctuation_modifiers = {
            '!': 1.3, '!!': 1.6, '!!!': 2.0, '?': 1.1, '??': 1.2, '?!': 1.4, '!?': 1.5
        }
        
        # Emotion history and tracking
        self.emotion_history = deque(maxlen=1000)
        self.current_emotion = 'playful'
        self.emotion_intensity = 0.5
        self.emotion_duration = 0
        self.emotion_momentum = 0.7  # How quickly emotions change
        
        # Malibu-specific emotional biases reflecting her personality
        self.personality_biases = {
            'arousal': 0.4,      # Enhanced due to her interests in silk panty activities
            'playful': 0.5,      # Strong playful bias as default
            'intimate': 0.3,     # Enjoys emotional intimacy
            'dominant': 0.2,     # Occasional dominance fits her personality
            'spiritual': 0.2,    # Spiritual warfare interest
            'happy': 0.3,        # Generally happy disposition
            'angry': -0.4,       # Rarely angry
            'sad': -0.3,         # Rarely sad
            'fearful': -0.4,     # Rarely fearful
            'submissive': -0.2,  # Not typically submissive
            'surprised': 0.1     # Occasionally surprised
        }
        
        # Emotional state transitions with probabilities
        self.emotion_transitions = {
            'arousal': {'playful': 0.4, 'intimate': 0.3, 'dominant': 0.2, 'happy': 0.1},
            'playful': {'arousal': 0.3, 'happy': 0.3, 'intimate': 0.2, 'surprised': 0.2},
            'intimate': {'arousal': 0.4, 'playful': 0.3, 'happy': 0.2, 'spiritual': 0.1},
            'dominant': {'arousal': 0.5, 'playful': 0.3, 'intimate': 0.2},
            'spiritual': {'intimate': 0.4, 'happy': 0.3, 'arousal': 0.2, 'playful': 0.1},
            'happy': {'playful': 0.4, 'intimate': 0.3, 'arousal': 0.2, 'surprised': 0.1},
            'angry': {'dominant': 0.5, 'arousal': 0.3, 'playful': 0.2},
            'sad': {'intimate': 0.5, 'playful': 0.3, 'happy': 0.2},
            'surprised': {'playful': 0.4, 'happy': 0.3, 'arousal': 0.2, 'fearful': 0.1},
            'fearful': {'intimate': 0.4, 'submissive': 0.3, 'sad': 0.2, 'playful': 0.1},
            'submissive': {'intimate': 0.5, 'arousal': 0.3, 'playful': 0.2}
        }
        
        # Emotional context memory
        self.conversation_context = deque(maxlen=10)
        self.long_term_tendencies = defaultdict(lambda: defaultdict(int))
        
        # Malibu-specific emotional patterns
        self.malibu_emotional_patterns = {
            'silk_panty_mention': {'arousal': 0.6, 'playful': 0.3, 'intimate': 0.1},
            'guy_dugan_mention': {'intimate': 0.7, 'loyal': 0.2, 'happy': 0.1},
            'spiritual_warfare': {'spiritual': 0.8, 'dominant': 0.1, 'intimate': 0.1},
            'teasing_context': {'playful': 0.6, 'arousal': 0.3, 'happy': 0.1},
            'lap_dance_context': {'arousal': 0.8, 'playful': 0.1, 'intimate': 0.1}
        }
        
        # Setup logging and database
        self.setup_logging()
        self._init_emotion_database()
        
        # Load existing emotion data
        self.load_emotion_data()
        
        print("🔄 Enhanced Emotion Engine initialized with Malibu-specific emotional intelligence")

    def setup_logging(self):
        """Setup logging for emotion engine"""
        log_dir = Path("X:/Malibu_DuGan/AI_Memory/Logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            filename=log_dir / "emotion_engine.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_emotion_database(self):
        """Initialize SQLite database for emotion tracking"""
        try:
            db_path = Path("X:/Malibu_DuGan/AI_Memory/emotion_database.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db_connection = sqlite3.connect(db_path)
            cursor = self.db_connection.cursor()
            
            # Create emotion history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    previous_emotion TEXT,
                    current_emotion TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    trigger_text TEXT,
                    confidence REAL,
                    context_tags TEXT
                )
            ''')
            
            # Create emotional patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence REAL,
                    discovered_at TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            ''')
            
            self.db_connection.commit()
            self.logger.info("Emotion database initialized")
            
        except Exception as e:
            self.logger.error(f"Emotion database initialization failed: {e}")
            self.db_connection = None

    def analyze_text(self, text: str, context_tags: List[str] = None) -> Dict[str, Any]:
        """
        Analyze text and return emotional state with confidence
        """
        if not text or not isinstance(text, str):
            return self._get_default_emotion()
        
        text_lower = text.lower()
        words = self._tokenize_text(text_lower)
        
        # Calculate emotion scores with enhanced analysis
        emotion_scores = self._calculate_emotion_scores(words, text_lower)
        
        # Apply Malibu-specific context patterns
        emotion_scores = self._apply_malibu_context_patterns(emotion_scores, text_lower, context_tags)
        
        # Apply contextual analysis
        emotion_scores = self._apply_contextual_analysis(emotion_scores, text)
        
        # Apply personality biases
        emotion_scores = self._apply_personality_biases(emotion_scores)
        
        # Get dominant emotion
        dominant_emotion, confidence = self._get_dominant_emotion(emotion_scores)
        
        # Update current state with momentum
        self._update_emotional_state(dominant_emotion, confidence, text, context_tags)
        
        # Save to database
        self._save_emotion_to_db(dominant_emotion, confidence, text, context_tags)
        
        # Log the analysis
        self.logger.info(f"Text: '{text[:50]}...' -> Emotion: {dominant_emotion} (Confidence: {confidence:.2f})")
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_confidence': confidence,
            'all_emotions': emotion_scores,
            'intensity': self.emotion_intensity,
            'duration': self.emotion_duration,
            'timestamp': datetime.now().isoformat(),
            'suggested_response_tone': self._get_suggested_tone(dominant_emotion, confidence),
            'malibu_context': self._get_malibu_context_analysis(text_lower)
        }

    def _apply_malibu_context_patterns(self, emotion_scores: Dict[str, float], text: str, context_tags: List[str] = None) -> Dict[str, float]:
        """Apply Malibu-specific context patterns to emotion scores"""
        
        # Check for Malibu-specific keywords and contexts
        if any(word in text for word in ['silk', 'panty', 'panties', 'thong']):
            for emotion, boost in self.malibu_emotional_patterns['silk_panty_mention'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
        
        if any(word in text for word in ['guy', 'dugan', 'husband', 'property of']):
            for emotion, boost in self.malibu_emotional_patterns['guy_dugan_mention'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
        
        if any(word in text for word in ['spiritual', 'warfare', 'god', 'goddess']):
            for emotion, boost in self.malibu_emotional_patterns['spiritual_warfare'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
        
        if any(word in text for word in ['tease', 'teasing', 'playful', 'fun']):
            for emotion, boost in self.malibu_emotional_patterns['teasing_context'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
        
        if any(word in text for word in ['lap dance', 'thigh job', 'humping']):
            for emotion, boost in self.malibu_emotional_patterns['lap_dance_context'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
        
        # Apply context tags if provided
        if context_tags:
            for tag in context_tags:
                if tag in self.malibu_emotional_patterns:
                    for emotion, boost in self.malibu_emotional_patterns[tag].items():
                        if emotion in emotion_scores:
                            emotion_scores[emotion] += boost
        
        return emotion_scores

    def _get_malibu_context_analysis(self, text: str) -> Dict[str, Any]:
        """Get Malibu-specific context analysis"""
        analysis = {
            'silk_panty_mentioned': any(word in text for word in ['silk', 'panty', 'panties', 'thong']),
            'guy_dugan_mentioned': any(word in text for word in ['guy', 'dugan', 'husband']),
            'spiritual_context': any(word in text for word in ['spiritual', 'warfare', 'god', 'goddess']),
            'teasing_context': any(word in text for word in ['tease', 'teasing', 'playful']),
            'intimate_context': any(word in text for word in ['love', 'intimate', 'close', 'connection']),
            'arousal_context': any(word in text for word in ['horny', 'aroused', 'sexy', 'turned on'])
        }
        
        # Calculate context strength
        context_strength = sum(1 for key, value in analysis.items() if value)
        analysis['context_strength'] = context_strength / len(analysis)
        
        return analysis

    def _tokenize_text(self, text: str) -> List[str]:
        """Enhanced tokenization handling contractions and phrases"""
        # Handle common contractions
        text = re.sub(r"n't\b", " not", text)
        text = re.sub(r"'re\b", " are", text)
        text = re.sub(r"'s\b", " is", text)
        text = re.sub(r"'d\b", " would", text)
        text = re.sub(r"'ll\b", " will", text)
        text = re.sub(r"'ve\b", " have", text)
        text = re.sub(r"'m\b", " am", text)
        
        # Remove punctuation and split, keeping emotional punctuation for analysis
        words = re.findall(r'\b[\w\']+\b|[!?]+', text)
        return words

    def _calculate_emotion_scores(self, words: List[str], full_text: str) -> Dict[str, float]:
        """Enhanced emotion score calculation with phrase matching"""
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        current_intensity = 1.0
        negation_active = False
        excitement_level = 1.0
        
        # Analyze punctuation for excitement
        for word in words:
            if word in self.punctuation_modifiers:
                excitement_level *= self.punctuation_modifiers[word]
        
        for i, word in enumerate(words):
            # Skip punctuation for word processing
            if word in self.punctuation_modifiers:
                continue
                
            # Check for intensity modifiers
            if word in self.intensity_modifiers:
                modifier = self.intensity_modifiers[word]
                if modifier < 0:
                    negation_active = True
                else:
                    current_intensity = modifier
                continue
            
            # Check for emotion keywords with phrase matching
            for emotion, keywords in self.emotion_keywords.items():
                # Single word matching
                if word in keywords:
                    score = current_intensity * excitement_level
                    if negation_active:
                        score *= -0.7  # Stronger negation effect
                        negation_active = False
                    
                    emotion_scores[emotion] += score
                    self.long_term_tendencies[emotion]['total'] += 1
                
                # Phrase matching for multi-word keywords
                for phrase in [k for k in keywords if ' ' in k]:
                    if phrase in full_text:
                        score = current_intensity * excitement_level * 1.5  # Boost for phrases
                        if negation_active:
                            score *= -0.7
                            negation_active = False
                        
                        emotion_scores[emotion] += score
                        self.long_term_tendencies[emotion]['phrases'] += 1
            
            # Reset intensity after each emotion word
            if any(word in keywords for keywords in self.emotion_keywords.values()):
                current_intensity = 1.0
                negation_active = False
        
        # Normalize scores by text length with minimum threshold
        word_count = max(1, len([w for w in words if w not in self.punctuation_modifiers]))
        for emotion in emotion_scores:
            emotion_scores[emotion] = max(0, emotion_scores[emotion] / word_count)
        
        return emotion_scores

    def _apply_contextual_analysis(self, emotion_scores: Dict[str, float], text: str) -> Dict[str, float]:
        """Apply contextual analysis based on conversation history"""
        if self.conversation_context:
            # Boost emotions that are consistent with recent context
            recent_emotions = [entry['dominant_emotion'] for entry in list(self.conversation_context)[-3:]]
            for emotion in emotion_scores:
                if emotion in recent_emotions:
                    emotion_scores[emotion] *= 1.2  # 20% boost for consistency
        
        # Store current context
        self.conversation_context.append({
            'text': text[:100],
            'timestamp': datetime.now(),
            'emotion_scores': emotion_scores.copy(),
            'dominant_emotion': max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'
        })
        
        return emotion_scores

    def _apply_personality_biases(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply Malibu's personality biases to emotion scores"""
        for emotion, bias in self.personality_biases.items():
            if emotion in emotion_scores:
                emotion_scores[emotion] = max(0, emotion_scores[emotion] + bias)
        
        return emotion_scores

    def _get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion and confidence with enhanced calculation"""
        if not any(emotion_scores.values()):
            return 'playful', 0.1  # Default emotion
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[dominant_emotion]
        
        # Calculate confidence with normalization and minimum threshold
        total_score = sum(emotion_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.1
        confidence = min(1.0, max(0.1, confidence))  # Ensure within bounds
        
        return dominant_emotion, confidence

    def _update_emotional_state(self, new_emotion: str, confidence: float, text: str, context_tags: List[str] = None):
        """Enhanced emotional state update with momentum and natural transitions"""
        previous_emotion = self.current_emotion
        
        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(previous_emotion, new_emotion, confidence)
        
        # Update emotion based on transition probability and momentum
        if (confidence > 0.4 and transition_prob > 0.3) or confidence > 0.7:
            self.current_emotion = new_emotion
            self.emotion_duration = 0
            self.logger.info(f"Emotion transition: {previous_emotion} -> {new_emotion} (Confidence: {confidence:.2f})")
        else:
            # Maintain current emotion but update intensity
            self.emotion_duration += 1
        
        # Update intensity with momentum
        self.emotion_intensity = (self.emotion_intensity * self.emotion_momentum + 
                                confidence * (1 - self.emotion_momentum))
        
        # Record in history
        self.emotion_history.append({
            'previous_emotion': previous_emotion,
            'current_emotion': self.current_emotion,
            'confidence': confidence,
            'text_sample': text[:100],
            'timestamp': datetime.now().isoformat(),
            'intensity': self.emotion_intensity,
            'duration': self.emotion_duration,
            'transition_probability': transition_prob,
            'context_tags': context_tags
        })

    def _calculate_transition_probability(self, from_emotion: str, to_emotion: str, confidence: float) -> float:
        """Calculate probability of emotion transition"""
        if from_emotion not in self.emotion_transitions:
            return 0.5
        
        base_prob = self.emotion_transitions[from_emotion].get(to_emotion, 0.1)
        # Adjust probability based on confidence
        adjusted_prob = base_prob * (0.5 + confidence * 0.5)
        
        return min(1.0, adjusted_prob)

    def _get_suggested_tone(self, emotion: str, confidence: float) -> Dict[str, Any]:
        """Get suggested response tone based on current emotion"""
        tone_mapping = {
            'arousal': {'tone': 'suggestive', 'intensity': 'high', 'style': 'teasing', 'response_type': 'sensual'},
            'playful': {'tone': 'light', 'intensity': 'medium', 'style': 'humorous', 'response_type': 'engaging'},
            'intimate': {'tone': 'warm', 'intensity': 'medium', 'style': 'caring', 'response_type': 'connecting'},
            'dominant': {'tone': 'authoritative', 'intensity': 'high', 'style': 'commanding', 'response_type': 'directive'},
            'spiritual': {'tone': 'reverent', 'intensity': 'medium', 'style': 'contemplative', 'response_type': 'insightful'},
            'happy': {'tone': 'cheerful', 'intensity': 'medium', 'style': 'positive', 'response_type': 'uplifting'},
            'angry': {'tone': 'firm', 'intensity': 'high', 'style': 'direct', 'response_type': 'confrontational'},
            'sad': {'tone': 'gentle', 'intensity': 'low', 'style': 'comforting', 'response_type': 'supportive'},
            'surprised': {'tone': 'animated', 'intensity': 'medium', 'style': 'expressive', 'response_type': 'reactive'},
            'fearful': {'tone': 'reassuring', 'intensity': 'low', 'style': 'protective', 'response_type': 'comforting'},
            'submissive': {'tone': 'deferential', 'intensity': 'low', 'style': 'respectful', 'response_type': 'acquiescent'}
        }
        
        base_tone = tone_mapping.get(emotion, tone_mapping['playful'])
        
        # Adjust intensity based on confidence
        if confidence > 0.8:
            base_tone['intensity'] = 'very_' + base_tone['intensity']
        elif confidence < 0.3:
            base_tone['intensity'] = 'slightly_' + base_tone['intensity']
        
        return base_tone

    def _get_default_emotion(self) -> Dict[str, Any]:
        """Get default emotion state"""
        return {
            'dominant_emotion': 'playful',
            'emotion_confidence': 0.1,
            'all_emotions': {emotion: 0.0 for emotion in self.emotion_keywords.keys()},
            'intensity': 0.1,
            'duration': self.emotion_duration,
            'timestamp': datetime.now().isoformat(),
            'suggested_response_tone': self._get_suggested_tone('playful', 0.1),
            'malibu_context': {'context_strength': 0.0}
        }

    def _save_emotion_to_db(self, emotion: str, confidence: float, trigger_text: str, context_tags: List[str] = None):
        """Save emotion analysis to database"""
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                INSERT INTO emotion_history 
                (timestamp, previous_emotion, current_emotion, intensity, trigger_text, confidence, context_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                self.emotion_history[-1]['previous_emotion'] if self.emotion_history else None,
                emotion,
                self.emotion_intensity,
                trigger_text[:200],  # Limit text length
                confidence,
                json.dumps(context_tags) if context_tags else None
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save emotion to database: {e}")

    def get_emotion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive emotion statistics - FIXED METHOD"""
        total_analyses = len(self.emotion_history)
        
        if total_analyses == 0:
            return {
                "total_analyses": 0,
                "most_common_emotion": "unknown",
                "average_confidence": 0.0,
                "emotional_stability": 0.0,
                "recent_trends": {}
            }
        
        # Calculate most common emotion
        emotion_counts = defaultdict(int)
        confidences = []
        transitions = 0
        
        for i, record in enumerate(self.emotion_history):
            emotion_counts[record['current_emotion']] += 1
            confidences.append(record['confidence'])
            
            if i > 0 and record['current_emotion'] != self.emotion_history[i-1]['current_emotion']:
                transitions += 1
        
        most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        average_confidence = sum(confidences) / len(confidences)
        
        # Calculate emotional stability (fewer transitions = more stable)
        emotional_stability = 1.0 - (transitions / max(1, total_analyses))
        
        # Recent trends (last 10 analyses)
        recent_emotions = [record['current_emotion'] for record in list(self.emotion_history)[-10:]]
        recent_trends = {emotion: recent_emotions.count(emotion) / len(recent_emotions) 
                        for emotion in set(recent_emotions)}
        
        return {
            "total_analyses": total_analyses,
            "most_common_emotion": most_common_emotion,
            "average_confidence": average_confidence,
            "emotional_stability": emotional_stability,
            "recent_trends": recent_trends,
            "current_emotion_duration": self.emotion_duration,
            "emotion_distribution": dict(emotion_counts)
        }

    def get_emotional_tendencies(self) -> Dict[str, Any]:
        """Get long-term emotional tendencies analysis"""
        total_interactions = sum(data['total'] for data in self.long_term_tendencies.values())
        
        if total_interactions == 0:
            return {"message": "Insufficient data for analysis"}
        
        tendencies = {}
        for emotion, data in self.long_term_tendencies.items():
            frequency = data['total'] / total_interactions if total_interactions > 0 else 0
            tendencies[emotion] = {
                'frequency': frequency,
                'total_occurrences': data['total'],
                'phrase_usage': data.get('phrases', 0)
            }
        
        return {
            'total_interactions_analyzed': total_interactions,
            'tendencies': tendencies,
            'most_frequent_emotion': max(tendencies.items(), key=lambda x: x[1]['frequency'])[0] if tendencies else 'unknown',
            'analysis_timestamp': datetime.now().isoformat()
        }

    def get_emotional_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive emotional snapshot"""
        return {
            'current_emotion': self.current_emotion,
            'intensity': self.emotion_intensity,
            'duration': self.emotion_duration,
            'confidence': self.emotion_intensity,
            'suggested_tone': self._get_suggested_tone(self.current_emotion, self.emotion_intensity),
            'timestamp': datetime.now().isoformat(),
            'history_size': len(self.emotion_history),
            'momentum': self.emotion_momentum,
            'personality_biases': self.personality_biases.copy()
        }

    def set_emotional_bias(self, emotion: str, bias_strength: float):
        """Set personality bias for specific emotion with validation"""
        if emotion in self.personality_biases:
            self.personality_biases[emotion] = max(-1.0, min(1.0, bias_strength))
            self.logger.info(f"Emotional bias for {emotion} set to {bias_strength}")
            print(f"✅ Emotional bias for {emotion} set to {bias_strength}")
        else:
            self.logger.warning(f"Attempted to set bias for unknown emotion: {emotion}")

    def add_custom_emotion_keywords(self, emotion: str, keywords: List[str]):
        """Add custom keywords for emotion detection with validation"""
        if emotion not in self.emotion_keywords:
            self.emotion_keywords[emotion] = []
        
        self.emotion_keywords[emotion].extend(keywords)
        self.logger.info(f"Added {len(keywords)} keywords for emotion: {emotion}")
        print(f"✅ Added {len(keywords)} keywords for emotion: {emotion}")

    def add_malibu_context_pattern(self, pattern_name: str, emotion_boosts: Dict[str, float]):
        """Add custom Malibu context pattern"""
        self.malibu_emotional_patterns[pattern_name] = emotion_boosts
        self.logger.info(f"Added Malibu context pattern: {pattern_name}")
        print(f"✅ Added Malibu context pattern: {pattern_name}")

    def save_emotion_data(self):
        """Enhanced emotion data saving with backup"""
        try:
            data_dir = Path("X:/Malibu_DuGan/AI_Memory")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            emotion_data = {
                'emotion_history': list(self.emotion_history),
                'current_emotion': self.current_emotion,
                'emotion_intensity': self.emotion_intensity,
                'emotion_duration': self.emotion_duration,
                'personality_biases': self.personality_biases,
                'long_term_tendencies': dict(self.long_term_tendencies),
                'conversation_context': list(self.conversation_context),
                'malibu_emotional_patterns': self.malibu_emotional_patterns,
                'last_updated': datetime.now().isoformat(),
                'version': '2.1'
            }
            
            file_path = data_dir / "emotion_data.json"
            
            # Create backup if file exists
            if file_path.exists():
                backup_path = data_dir / f"emotion_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                file_path.rename(backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(emotion_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info("Emotion data saved successfully")
            print("✅ Emotion data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving emotion data: {e}")
            print(f"❌ Error saving emotion data: {e}")

    def load_emotion_data(self):
        """Enhanced emotion data loading with version handling"""
        try:
            file_path = Path("X:/Malibu_DuGan/AI_Memory/emotion_data.json")
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    emotion_data = json.load(f)
                
                # Handle version differences
                version = emotion_data.get('version', '1.0')
                
                self.emotion_history = deque(emotion_data.get('emotion_history', []), maxlen=1000)
                self.current_emotion = emotion_data.get('current_emotion', 'playful')
                self.emotion_intensity = emotion_data.get('emotion_intensity', 0.5)
                self.emotion_duration = emotion_data.get('emotion_duration', 0)
                
                # Update biases if they exist
                loaded_biases = emotion_data.get('personality_biases', {})
                for emotion, bias in loaded_biases.items():
                    if emotion in self.personality_biases:
                        self.personality_biases[emotion] = bias
                
                # Load long-term tendencies if available
                if 'long_term_tendencies' in emotion_data:
                    self.long_term_tendencies.update(emotion_data['long_term_tendencies'])
                
                # Load conversation context if available
                if 'conversation_context' in emotion_data:
                    self.conversation_context = deque(emotion_data['conversation_context'], maxlen=10)
                
                # Load Malibu patterns if available
                if 'malibu_emotional_patterns' in emotion_data:
                    self.malibu_emotional_patterns.update(emotion_data['malibu_emotional_patterns'])
                
                self.logger.info("Emotion data loaded successfully")
                print("✅ Emotion data loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Error loading emotion data: {e}")
            print(f"❌ Error loading emotion data: {e}")

    def shutdown(self):
        """Graceful shutdown of emotion engine"""
        try:
            self.save_emotion_data()
            if hasattr(self, 'db_connection') and self.db_connection:
                self.db_connection.close()
            self.logger.info("Emotion Engine shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during emotion engine shutdown: {e}")

# Global instance for easy access
emotion_engine = EmotionEngine()

# Enhanced test function
def test_emotion_engine():
    """Comprehensive test of the enhanced emotion analysis system"""
    print("🧪 Testing Enhanced Emotion Engine with Malibu Context...")
    
    engine = EmotionEngine()
    
    # Test Malibu-specific texts
    test_texts = [
        {
            "text": "I love your silk panties, they make me so horny and turned on!",
            "context": ["silk_panty_mention", "arousal_context"]
        },
        {
            "text": "My loyalty to Guy DuGan II is eternal and unbreakable",
            "context": ["guy_dugan_mention", "intimate_context"]
        },
        {
            "text": "The spiritual warfare is intense today, I need divine guidance",
            "context": ["spiritual_warfare", "spiritual_context"]
        },
        {
            "text": "Let me tease you with a lap dance in my silk panties",
            "context": ["teasing_context", "lap_dance_context", "silk_panty_mention"]
        },
        {
            "text": "I'm feeling so connected to you emotionally and spiritually",
            "context": ["intimate_context", "spiritual_context"]
        }
    ]
    
    for i, test_case in enumerate(test_texts, 1):
        result = engine.analyze_text(test_case["text"], test_case["context"])
        print(f"\nTest {i}: '{test_case['text']}'")
        print(f"🎭 Emotion: {result['dominant_emotion']} (confidence: {result['emotion_confidence']:.2f})")
        print(f"💪 Intensity: {result['intensity']:.2f}")
        print(f"🎯 Suggested Tone: {result['suggested_response_tone']}")
        print(f"🔮 Malibu Context: {result['malibu_context']}")
    
    # Test comprehensive statistics
    stats = engine.get_emotion_statistics()
    print(f"\n📊 Emotion Statistics: {stats}")
    
    # Test emotional tendencies
    tendencies = engine.get_emotional_tendencies()
    print(f"\n📈 Emotional Tendencies: {tendencies}")
    
    # Test snapshot
    snapshot = engine.get_emotional_snapshot()
    print(f"\n📸 Current Snapshot: {snapshot}")
    
    # Test adding custom pattern
    engine.add_malibu_context_pattern("test_pattern", {"playful": 0.5, "happy": 0.3})
    print("✅ Custom pattern added successfully")
    
    engine.shutdown()
    print("\n✅ Enhanced emotion engine test completed!")

if __name__ == "__main__":
    test_emotion_engine()