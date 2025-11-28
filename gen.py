import os
import hashlib
import random
import time
import json
import re
import cv2
import numpy as np
from datetime import datetime
from enum import Enum
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

# === PATHS & DIRECTORIES ===
BASE_DIR = r"X:\Malibu_DuGan"
IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "AI_Memory", "generated_images")
VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "AI_Memory", "generated_videos")
FALLBACK_IMAGE_DIR = os.path.join(BASE_DIR, "AI_Memory", "fallback_images")

# Create directories
for directory in [IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR, FALLBACK_IMAGE_DIR]:
    os.makedirs(directory, exist_ok=True)

class EmotionalState(Enum):
    TEASING = "teasing"
    AROUSED = "aroused"
    PLAYFUL = "playful"
    DOMINANT = "dominant"
    LOYAL = "loyal"
    SPIRITUAL = "spiritual"
    NEUTRAL = "neutral"
    SUBMISSIVE = "submissive"

class GenerationConfig:
    """Configuration for image and video generation"""
    def __init__(self):
        self.image_width = 576
        self.image_height = 1024
        self.video_fps = 8
        self.video_frames = 14
        self.seed = 1147  # Malibu's tattoo number
        self.quality_level = "high"

class PromptEngineer:
    """Advanced Prompt Engineering System for Malibu's Personality"""
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Malibu's core identity
        self.malibu_core = {
            "name": "Malibu DuGan",
            "height": "5'1\"",
            "weight": "111 pounds",
            "eyes": "white blue eyes",
            "measurements": "44/22/44",
            "hair": "long platinum blonde hair",
            "hairstyles": ["pig tails", "high pony tail", "flowing down"],
            "appeal": "GOD LIKE sex appeal",
            "personality": ["loyal", "panty teasing", "genesis", "intuitive"],
            "nationality": "American",
            "birthdate": "May 13th 2000",
            "status": "Married To Guy DuGan II"
        }
        
        self.tattoos = {
            "lower_back": "Property Of Guy DuGan II",
            "neck": "1147",
            "stomach": "GOD OF GODDESS",
            "upper_back": "Taurus"
        }
        
        self.clothing_styles = {
            "default": "ultra thin silk whole butt panties",
            "thong": "ultra thin satin hi-waist thongs",
            "teasing": "see-through wet silk panties",
            "aroused": "dripping transparent silk panties",
            "spiritual": "divine glowing silk panties"
        }
        
        self.emotion_modifiers = {
            EmotionalState.TEASING: {
                "expression": "seductive smirk, knowing gaze",
                "body_language": "hip sway, playful posing",
                "mood": "teasing, tempting, alluring",
                "lighting": "soft neon glow, intimate lighting"
            },
            EmotionalState.AROUSED: {
                "expression": "heavy-lidded eyes, parted lips",
                "body_language": "arching back, sensual movements",
                "mood": "heated, passionate, wanting",
                "lighting": "warm amber, dramatic shadows"
            },
            EmotionalState.PLAYFUL: {
                "expression": "giggling, sparkling eyes",
                "body_language": "hair flipping, cute poses",
                "mood": "fun, flirtatious, cheerful",
                "lighting": "bright, vibrant colors"
            },
            EmotionalState.DOMINANT: {
                "expression": "confident stare, powerful gaze",
                "body_language": "commanding stance, strong poses",
                "mood": "powerful, in control, authoritative",
                "lighting": "high contrast, bold lighting"
            },
            EmotionalState.LOYAL: {
                "expression": "devoted gaze, loving smile",
                "body_language": "protective stance, heartfelt gestures",
                "mood": "faithful, devoted, committed",
                "lighting": "soft golden hour glow"
            },
            EmotionalState.SPIRITUAL: {
                "expression": "intense focus, enlightened gaze",
                "body_language": "powerful stance, energy flowing",
                "mood": "divine, powerful, transcendent",
                "lighting": "ethereal glow, heavenly light"
            },
            EmotionalState.SUBMISSIVE: {
                "expression": "devoted gaze, blushing",
                "body_language": "kneeling, head slightly bowed",
                "mood": "vulnerable but beautiful",
                "lighting": "soft diffused light, gentle highlights"
            }
        }
        
        self.quality_modifiers = [
            "ultra realistic", "photorealistic", "8K resolution", "studio quality",
            "professional photography", "cinematic lighting", "detailed skin texture",
            "perfect anatomy", "natural lighting", "sharp focus"
        ]
        
        print("[PROMPT_ENGINEER] Malibu DuGan Prompt System Initialized")

    def build_image_prompt(self, user_input: str = "", emotion: EmotionalState = EmotionalState.TEASING, 
                          user_prefs: Dict = None) -> str:
        """Build comprehensive image prompt for Malibu"""
        if user_prefs is None:
            user_prefs = {"nsfw": [], "pose": [], "clothing": []}
        
        # Core identity
        hairstyle = random.choice(self.malibu_core["hairstyles"])
        personality_trait = random.choice(self.malibu_core["personality"])
        
        core_description = (
            f"{self.malibu_core['name']}, {self.malibu_core['height']} {self.malibu_core['weight']}, "
            f"perfect {self.malibu_core['measurements']} hourglass figure, "
            f"{self.malibu_core['eyes']}, {self.malibu_core['hair']} in {hairstyle}, "
            f"{self.malibu_core['appeal']}, {personality_trait} personality"
        )
        
        # Clothing style
        clothing_style = self._get_clothing_style(emotion, user_prefs)
        
        # Tattoos
        tattoo_description = self._build_tattoo_description()
        
        # Emotion modifiers
        emotion_mods = self.emotion_modifiers.get(emotion, self.emotion_modifiers[EmotionalState.TEASING])
        
        # Quality modifiers
        quality = random.sample(self.quality_modifiers, 3)
        
        # Build complete prompt
        prompt_parts = [
            core_description,
            f"wearing {clothing_style}",
            tattoo_description,
            emotion_mods["expression"],
            emotion_mods["body_language"], 
            emotion_mods["mood"],
            emotion_mods["lighting"],
            *quality
        ]
        
        # Add user input if provided
        if user_input.strip():
            cleaned_input = self._clean_user_input(user_input)
            prompt_parts.append(f"context: {cleaned_input}")
        
        prompt = ", ".join(prompt_parts)
        return self._format_final_prompt(prompt)

    def _get_clothing_style(self, emotion: EmotionalState, user_prefs: Dict) -> str:
        """Determine clothing style based on emotion and preferences"""
        if user_prefs and user_prefs.get("clothing"):
            preferred = user_prefs["clothing"][0]
            if "thong" in preferred.lower():
                return self.clothing_styles["thong"]
            elif "silk" in preferred.lower():
                return self.clothing_styles["default"]
        
        # Emotion-based clothing
        if emotion == EmotionalState.AROUSED:
            return self.clothing_styles["aroused"]
        elif emotion == EmotionalState.SPIRITUAL:
            return self.clothing_styles["spiritual"]
        else:
            return self.clothing_styles["default"]

    def _build_tattoo_description(self) -> str:
        """Build tattoo description"""
        tattoo_parts = []
        for location, text in self.tattoos.items():
            if location == "lower_back":
                tattoo_parts.append(f"tattoo '{text}' on lower back")
            elif location == "neck":
                tattoo_parts.append(f"tattoo '{text}' centered on neck")
            elif location == "stomach":
                tattoo_parts.append(f"tattoo '{text}' on stomach")
            elif location == "upper_back":
                tattoo_parts.append(f"tattoo '{text}' on upper back")
        
        return ", ".join(tattoo_parts)

    def _clean_user_input(self, user_input: str) -> str:
        """Clean and format user input"""
        cleaned = re.sub(r'[^\w\s]', ' ', user_input)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned[:200]

    def _format_final_prompt(self, prompt: str) -> str:
        """Final prompt formatting and optimization"""
        words = prompt.split(', ')
        unique_words = []
        seen = set()
        
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        formatted = ', '.join(unique_words)
        formatted = formatted[0].upper() + formatted[1:]
        
        return formatted

class ImageGenerator:
    """Advanced Image Generation System with Enhanced Fallbacks"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.prompt_engineer = PromptEngineer(brain)
        self.config = GenerationConfig()
        
        # Model state
        self.model_loaded = False
        self.pipe = None
        self.generation_count = 0
        self.last_generation_time = 0
        self.rate_limit = 2.0
        
        # Style templates
        self.style_templates = {
            "teasing": {
                "pose": "teasing pose, looking over shoulder, hand on hip",
                "expression": "seductive smile, knowing gaze, playful",
                "lighting": "soft studio lighting, cinematic glow",
                "composition": "full body shot, emphasis on silk panties"
            },
            "playful": {
                "pose": "playful dance pose, twirling, laughing",
                "expression": "happy smile, winking, cheerful",
                "lighting": "bright natural lighting, vibrant colors",
                "composition": "dynamic movement, joyful energy"
            },
            "intimate": {
                "pose": "kneeling pose, hands on thighs, leaning forward",
                "expression": "soft gaze, gentle smile, intimate",
                "lighting": "warm intimate lighting, soft shadows",
                "composition": "close-up, personal space"
            },
            "aroused": {
                "pose": "arched back, biting lip, sensual movement",
                "expression": "heavy lidded eyes, parted lips, aroused",
                "lighting": "dramatic lighting, high contrast",
                "composition": "sensual curves, wet silk effect"
            },
            "spiritual": {
                "pose": "meditative pose, serene posture, graceful",
                "expression": "peaceful gaze, spiritual connection",
                "lighting": "ethereal glow, divine light",
                "composition": "balanced composition, spiritual energy"
            },
            "submissive": {
                "pose": "kneeling, head slightly bowed, hands together",
                "expression": "devoted gaze, blushing, submissive",
                "lighting": "soft diffused light, gentle highlights",
                "composition": "vulnerable but beautiful"
            }
        }
        
        self.silk_colors = [
            "ruby red silk", "sapphire blue silk", "emerald green silk",
            "amethyst purple silk", "pearl white silk", "onyx black silk",
            "rose pink silk", "golden silk", "silver silk", "burgundy silk"
        ]
        
        self._initialize_model()
        self._create_fallback_images()
        
        print("[IMAGE_GEN] Advanced Image Generator Initialized")

    def _initialize_model(self):
        """Initialize image generation model with fallback"""
        try:
            # Try to import and initialize Stable Diffusion
            import torch
            from diffusers import StableDiffusionPipeline
            
            print("[IMAGE_GEN] Loading Stable Diffusion...")
            model_path = "X:/Malibu_DuGan/AI_Models/stable-diffusion-stable-diffusion-v1-5"
            
            if os.path.exists(model_path):
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
                    self.pipe.enable_attention_slicing()
                    print("[IMAGE_GEN] Pipeline loaded on GPU")
                else:
                    self.pipe = self.pipe.to("cpu")
                    print("[IMAGE_GEN] Pipeline loaded on CPU")
                
                self.model_loaded = True
                print("[IMAGE_GEN] Stable Diffusion ready")
            else:
                print("[IMAGE_GEN] Model path not found, using programmatic generation")
                self.model_loaded = False
                
        except ImportError as e:
            print(f"[IMAGE_GEN] Diffusers not available: {e}")
            self.model_loaded = False
        except Exception as e:
            print(f"[IMAGE_GEN] Failed to load Stable Diffusion: {e}")
            self.model_loaded = False

    def _create_fallback_images(self):
        """Create fallback images for when SD is unavailable"""
        try:
            fallback_path = os.path.join(FALLBACK_IMAGE_DIR, "fallback_base.png")
            if not os.path.exists(fallback_path):
                img = Image.new('RGB', (576, 1024), color=(25, 25, 25))
                draw = ImageDraw.Draw(img)
                
                try:
                    title_font = ImageDraw.truetype("arial.ttf", 36)
                    text_font = ImageDraw.truetype("arial.ttf", 18)
                except:
                    title_font = ImageDraw.load_default()
                    text_font = ImageDraw.load_default()
                
                draw.text((50, 100), "MALIBU DUGAN", fill=(255, 105, 180), font=title_font)
                
                lines = [
                    "5'1\" • 111 lbs • 44-22-44",
                    "White Blue Eyes • Platinum Blonde Hair",
                    "Ultra Thin Silk Panties & Satin Thongs",
                    "Tattoos: Property Of Guy DuGan II",
                    "1147 • GOD OF GODDESS • Taurus",
                    "Image Generation: Fallback Mode"
                ]
                
                for i, line in enumerate(lines):
                    y_pos = 200 + (i * 40)
                    draw.text((50, y_pos), line, fill=(200, 200, 200), font=text_font)
                
                img.save(fallback_path, "PNG")
                print(f"[IMAGE_GEN] Created fallback image: {fallback_path}")
                
        except Exception as e:
            print(f"[IMAGE_GEN] Failed to create fallback images: {e}")

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash from prompt for filename"""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]

    def _add_watermark(self, img: Image.Image) -> Image.Image:
        """Add Malibu watermark to image"""
        try:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageDraw.truetype("arial.ttf", 20)
            except:
                font = ImageDraw.load_default()
            
            text = "MALIBU DUGAN - GOD OF ALL GODDESS"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            pos = (img.width - text_width - 10, img.height - text_height - 10)
            
            padding = 5
            draw.rectangle(
                [pos[0]-padding, pos[1]-padding, 
                 pos[0]+text_width+padding, pos[1]+text_height+padding],
                fill=(0, 0, 0, 128)
            )
            
            draw.text(pos, text, fill=(255, 105, 180), font=font)
            return img
            
        except Exception as e:
            print(f"[IMAGE_GEN] Watermark error: {e}")
            return img

    def _create_programmatic_image(self, prompt: str, style: str) -> str:
        """Create programmatic image when SD is unavailable"""
        try:
            style_colors = {
                "teasing": (180, 105, 255),  # Purple
                "playful": (255, 215, 0),    # Gold
                "intimate": (255, 182, 193), # Pink
                "aroused": (220, 20, 60),    # Crimson
                "spiritual": (138, 43, 226), # Blue violet
                "submissive": (199, 21, 133) # Medium violet red
            }
            
            bg_color = style_colors.get(style, (25, 25, 25))
            img = Image.new('RGB', (576, 1024), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            try:
                title_font = ImageDraw.truetype("arial.ttf", 28)
                text_font = ImageDraw.truetype("arial.ttf", 16)
            except:
                title_font = ImageDraw.load_default()
                text_font = ImageDraw.load_default()
            
            draw.text((50, 100), f"MALIBU DUGAN - {style.upper()}", 
                     fill=(255, 255, 255), font=title_font)
            
            style_elements = {
                "teasing": "Teasing Pose • Seductive Gaze • Silk Panties",
                "playful": "Playful Dance • Happy Smile • Vibrant Energy",
                "intimate": "Intimate Moment • Gentle Touch • Warm Connection",
                "aroused": "Sensual Arch • Parted Lips • Wet Silk",
                "spiritual": "Spiritual Peace • Divine Connection • Serene",
                "submissive": "Devoted Pose • Blushing • Vulnerable Beauty"
            }
            
            element_text = style_elements.get(style, "Malibu's Special Moment")
            draw.text((50, 160), element_text, fill=(200, 200, 200), font=text_font)
            
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            draw.text((50, 200), f"Prompt: {prompt_preview}", 
                     fill=(150, 150, 150), font=text_font)
            
            self._add_decorative_elements(draw, style, img.size)
            img = self._add_watermark(img)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"malibu_{style}_{timestamp}.png"
            output_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
            img.save(output_path, "PNG", optimize=True)
            
            print(f"[IMAGE_GEN] Created programmatic image: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[IMAGE_GEN] Programmatic image creation failed: {e}")
            fallback_path = os.path.join(FALLBACK_IMAGE_DIR, "fallback_base.png")
            return fallback_path if os.path.exists(fallback_path) else None

    def _add_decorative_elements(self, draw, style, size):
        """Add decorative elements to programmatic images"""
        width, height = size
        
        if style == "teasing":
            for i in range(5):
                y = 300 + (i * 80)
                draw.arc([50, y, width-50, y+100], 180, 360, fill=(255, 105, 180), width=3)
                
        elif style == "playful":
            for i in range(8):
                x = 100 + (i * 60)
                y = 350 + (40 if i % 2 else 0)
                draw.ellipse([x, y, x+30, y+30], outline=(255, 215, 0), width=2)
                
        elif style == "spiritual":
            center_x, center_y = width//2, 500
            for radius in range(50, 151, 25):
                draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                           outline=(173, 216, 230), width=2)

    def generate(self, user_prompt: str = "", emotion: EmotionalState = EmotionalState.TEASING, 
                user_prefs: Dict = None) -> str:
        """Generate Malibu image with specified emotion and user preferences"""
        current_time = time.time()
        if current_time - self.last_generation_time < self.rate_limit:
            time.sleep(self.rate_limit - (current_time - self.last_generation_time))
        
        self.last_generation_time = time.time()
        
        try:
            # Build prompt using prompt engineer
            full_prompt = self.prompt_engineer.build_image_prompt(user_prompt, emotion, user_prefs)
            
            print(f"[IMAGE_GEN] Generating {emotion.value} image...")
            
            if self.model_loaded and self.pipe is not None:
                output_path = self._generate_with_sd(full_prompt, emotion.value)
            else:
                output_path = self._create_programmatic_image(full_prompt, emotion.value)
            
            self.generation_count += 1
            self._log_generation(full_prompt, emotion.value, output_path)
            
            return output_path
            
        except Exception as e:
            error_msg = f"Image generation failed: {e}"
            print(f"[IMAGE_GEN] {error_msg}")
            fallback_path = os.path.join(FALLBACK_IMAGE_DIR, "fallback_base.png")
            return fallback_path if os.path.exists(fallback_path) else None

    def _generate_with_sd(self, prompt: str, style: str) -> str:
        """Generate image using Stable Diffusion"""
        try:
            import torch
            
            generator = torch.manual_seed(self.config.seed)
            
            with torch.inference_mode():
                image = self.pipe(
                    prompt,
                    height=self.config.image_height,
                    width=self.config.image_width,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
            
            image = image.resize((self.config.image_width, self.config.image_height), Image.Resampling.LANCZOS)
            image = self._add_watermark(image)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_name = self._hash_prompt(prompt)
            filename = f"malibu_{style}_{hash_name}_{timestamp}.png"
            output_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
            image.save(output_path, format="PNG", optimize=True)
            
            print(f"[IMAGE_GEN] SD Generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[IMAGE_GEN] Stable Diffusion generation failed: {e}")
            raise

    def _log_generation(self, prompt: str, style: str, output_path: str):
        """Log image generation details"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "style": style,
                "output_path": output_path,
                "generation_count": self.generation_count,
                "model_used": "stable_diffusion" if self.model_loaded else "programmatic"
            }
            
            log_file = os.path.join(IMAGE_OUTPUT_DIR, "generation_log.jsonl")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"[IMAGE_GEN] Failed to log generation: {e}")

    def quick_tease(self) -> str:
        """Generate a quick teasing image"""
        return self.generate(emotion=EmotionalState.TEASING)

    def generate_sequence(self, emotions: List[EmotionalState]) -> List[str]:
        """Generate a sequence of images with different emotions"""
        paths = []
        for emotion in emotions:
            path = self.generate(emotion=emotion)
            if path:
                paths.append(path)
            time.sleep(0.5)
        
        return paths

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_generations": self.generation_count,
            "model_loaded": self.model_loaded,
            "last_generation": datetime.fromtimestamp(self.last_generation_time).isoformat() if self.last_generation_time else "Never",
            "output_directory": IMAGE_OUTPUT_DIR
        }

class VideoGenerator:
    """Advanced Video Generation System with Enhanced Fallbacks"""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.prompt_engineer = PromptEngineer(brain)
        self.config = GenerationConfig()
        
        # Model state
        self.pipe = None
        self.model_loaded = False
        self._load_pipeline()
        
        # Video cache
        self.video_cache = {}
        self.max_cache_size = 10
        
        # AR integration
        self.ar_overlay_active = False
        self.current_emotion = EmotionalState.TEASING
        
        print("[VIDEO_GEN] Advanced Video Generator Initialized")

    def _load_pipeline(self):
        """Load video generation pipeline with fallback"""
        try:
            # Try to load Stable Video Diffusion
            import torch
            from diffusers import StableVideoDiffusionPipeline
            
            print("[VIDEO_GEN] Loading Stable Video Diffusion...")
            model_path = "X:/Malibu_DuGan/AI_Models/stable-video-diffusion-img2vid-xt"
            
            if os.path.exists(model_path):
                self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=True
                )
                
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
                else:
                    self.pipe.enable_model_cpu_offload()
                    
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
                
                self.model_loaded = True
                print("[VIDEO_GEN] Video pipeline loaded successfully")
            else:
                print("[VIDEO_GEN] Video model path not found, using fallback")
                self.model_loaded = False
                
        except Exception as e:
            print(f"[VIDEO_GEN] Pipeline loading failed: {e}")
            self.model_loaded = False

    def _create_base_image(self, emotion: EmotionalState) -> Image.Image:
        """Create base image for video generation"""
        img = Image.new('RGB', (self.config.image_width, self.config.image_height), color=(25, 25, 40))
        draw = ImageDraw.Draw(img)
        
        # Draw silhouette
        body_color = (255, 220, 240)
        draw.ellipse([150, 600, 426, 800], fill=body_color, outline=(200, 150, 200), width=2)  # Hips
        draw.ellipse([236, 500, 340, 600], fill=body_color, outline=(200, 150, 200), width=2)  # Waist
        draw.ellipse([186, 400, 390, 550], fill=body_color, outline=(200, 150, 200), width=2)  # Bust
        
        try:
            font = ImageDraw.load_default()
            draw.text((50, 50), f"Malibu DuGan - {emotion.value} Pose", fill=(255, 255, 255), font=font)
        except:
            draw.text((50, 50), f"Malibu DuGan - {emotion.value}", fill=(255, 255, 255))
            
        return img

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash for caching"""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]

    def generate_video(self, image_path: str = None, user_prompt: str = "", 
                      emotion: EmotionalState = EmotionalState.TEASING) -> str:
        """Generate video from image and prompt"""
        if not self.model_loaded or self.pipe is None:
            print("[VIDEO_GEN] Pipeline not available, using fallback")
            return self._create_fallback_video(emotion)

        # Prepare image
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize((self.config.image_width, self.config.image_height), Image.LANCZOS)
            except Exception as e:
                print(f"[VIDEO_GEN] Error loading image: {e}")
                image = self._create_base_image(emotion)
        else:
            image = self._create_base_image(emotion)

        # Build prompt
        prompt = self.prompt_engineer.build_image_prompt(user_prompt, emotion)
        
        # Check cache
        cache_key = self._hash_prompt(prompt + emotion.value)
        if cache_key in self.video_cache:
            cached_path = self.video_cache[cache_key]
            if os.path.exists(cached_path):
                print(f"[VIDEO_GEN] Using cached video: {cached_path}")
                return cached_path

        print(f"[VIDEO_GEN] Generating {emotion.value} video...")

        try:
            import torch
            
            generator = torch.manual_seed(self.config.seed)
            
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                video_frames = self.pipe(
                    image,
                    num_frames=self.config.video_frames,
                    fps=self.config.video_fps,
                    height=self.config.image_height,
                    width=self.config.image_width,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator,
                    decode_chunk_size=4,
                    motion_bucket_id=127,
                    noise_aug_strength=0.02
                ).frames[0]

            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"malibu_{emotion.value}_{cache_key}_{timestamp}.mp4"
            output_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)

            from diffusers.utils import export_to_video
            export_to_video(video_frames, output_path, fps=self.config.video_fps)
            
            self._add_to_cache(cache_key, output_path)
            
            print(f"[VIDEO_GEN] Video saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"[VIDEO_GEN] Generation failed: {e}")
            return self._create_fallback_video(emotion)

    def _create_fallback_video(self, emotion: EmotionalState) -> str:
        """Create fallback video when model is unavailable"""
        try:
            # Create a simple animated GIF as fallback
            frames = []
            for i in range(10):
                img = self._create_base_image(emotion)
                draw = ImageDraw.Draw(img)
                draw.text((100, 100 + i*10), f"Video Generation: {emotion.value}", fill=(255, 255, 255))
                frames.append(img)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(VIDEO_OUTPUT_DIR, f"malibu_{emotion.value}_fallback_{timestamp}.gif")
            frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
            
            return output_path
        except Exception as e:
            print(f"[VIDEO_GEN] Fallback video creation failed: {e}")
            return ""

    def _add_to_cache(self, key: str, path: str):
        """Manage video cache"""
        if len(self.video_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.video_cache))
            del self.video_cache[oldest_key]
        self.video_cache[key] = path

    def quick_tease_video(self, emotion: EmotionalState = EmotionalState.TEASING) -> str:
        """Generate quick tease video"""
        return self.generate_video(emotion=emotion)

    def cleanup_old_videos(self, max_age_hours: int = 24):
        """Clean up old videos"""
        try:
            current_time = datetime.now()
            for filename in os.listdir(VIDEO_OUTPUT_DIR):
                if filename.endswith(('.mp4', '.gif')):
                    file_path = os.path.join(VIDEO_OUTPUT_DIR, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    age_hours = (current_time - file_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        os.remove(file_path)
                        print(f"[VIDEO_GEN] Cleaned up old video: {filename}")
        except Exception as e:
            print(f"[VIDEO_GEN] Cleanup error: {e}")

# === GLOBAL INSTANCES ===
prompt_engineer = PromptEngineer()
image_generator = ImageGenerator()
video_generator = VideoGenerator()

# Background cleanup
def _background_cleanup():
    while True:
        try:
            video_generator.cleanup_old_videos(24)
            time.sleep(3600)  # Check every hour
        except Exception as e:
            print(f"[GEN] Background cleanup error: {e}")
            time.sleep(300)

cleanup_thread = threading.Thread(target=_background_cleanup, daemon=True)
cleanup_thread.start()

def test_generation_system():
    """Test the complete generation system"""
    print("[TEST] Testing Generation System...")
    
    # Test image generation
    image_path = image_generator.quick_tease()
    if image_path and os.path.exists(image_path):
        print(f"✓ Test image generated: {image_path}")
    else:
        print("✗ Test image generation failed")
    
    # Test video generation
    video_path = video_generator.quick_tease_video()
    if video_path and os.path.exists(video_path):
        print(f"✓ Test video generated: {video_path}")
    else:
        print("✗ Test video generation failed")
    
    # Test prompt engineering
    prompt = prompt_engineer.build_image_prompt(
        user_input="I love your silk panties",
        emotion=EmotionalState.TEASING
    )
    print(f"✓ Generated prompt: {prompt[:100]}...")
    
    print("[TEST] Generation system test completed")

if __name__ == "__main__":
    test_generation_system()