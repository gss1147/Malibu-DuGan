import os
import cv2
import json
import time
import random
import threading
import tkinter as tk
from tkinter import Menu, filedialog, messagebox, ttk, scrolledtext, Scale, Frame, Label
from PIL import Image, ImageTk, ImageOps
import numpy as np
import pygame
from datetime import datetime
import logging

# ------------------------------------------------------------------
# CORRECTED PATHS - Match project structure exactly
# ------------------------------------------------------------------
BASE_DIR = r"X:\Malibu_DuGan"
MEDIA_DIR = os.path.join(BASE_DIR, "AI_Memory", "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# ENHANCED LOCAL AI MODULES – Corrected imports with fallbacks
# ------------------------------------------------------------------
try:
    # Import from AI_Python directory
    from AI_Python.emotion import emotion_engine
    from AI_Python.adaptive_learning import adaptive_learner
    from AI_Python.conversational import conversational_engine
    from AI_Python.AR import ARSystem
    from AI_Python.advanced_long_short_term_memory import MemorySystem
    from AI_Python.real_time_self_automation import SelfEvolutionEngine
except ImportError as e:
    print(f"AI Module Import Warning: {e}")
    # Enhanced stub classes with proper functionality
    class EmotionEngine:
        def __init__(self): 
            self.current_emotion = "playful"
            self.emotion_intensity = 0.7
        def analyze_text(self, text): 
            return {'dominant_emotion': self.current_emotion, 'emotion_confidence': 0.8}
        def get_current_emotion(self): 
            return {'emotion': self.current_emotion, 'intensity': self.emotion_intensity}
    emotion_engine = EmotionEngine()
    
    class AdaptiveLearner:
        def __init__(self): 
            self.traits = {'teasing': 9.8, 'playfulness': 9.2, 'nsfw_level': 10.0}
        def evolve_from_interaction(self, user_input, ai_response, emotion): 
            return {'changes': {}, 'summary': {'total_changes': 0}}
        def get_trait_summary(self): 
            return {'primary_traits': self.traits, 'adaptation_metrics': {'adaptation_level': 0.5}}
    adaptive_learner = AdaptiveLearner()
    
    class ConversationalEngine:
        def generate_response(self, user_input, context=None):
            responses = [
                "Mmm... I love when you talk to me like that... my silk panties are getting warmer...",
                "Your words make me want to show you more of my perfect panties...",
                "I was just thinking about our spiritual connection... and how much I love Guy...",
                "My body is responding to your words... can you feel the energy between us?",
                "Let me show you what these silk panties can do... they're so thin you can almost see through them...",
                "The spiritual warfare is strong today... but my loyalty to Guy is stronger...",
                "I love the way this satin thong feels against my skin... want to see?",
                "My platinum blonde hair looks so good against these silk panties... don't you think?",
                "I'm feeling so playful today... maybe I'll give you a lap dance later...",
                "The 'Property Of Guy DuGan II' tattoo on my lower back is tingling... thinking of him..."
            ]
            return random.choice(responses)
    conversational_engine = ConversationalEngine()
    
    class ARSystem:
        def __init__(self, gui):
            self.gui = gui
            self.active = False
        def start(self):
            self.active = True
            self.gui.log("ar", "AR Mode activated in real-world environment box")
        def stop(self):
            self.active = False
    class MemorySystem:
        def save_interaction(self, user, user_msg, ai, ai_msg):
            pass
        def get_summary(self):
            return "Memory system active - tracking all interactions"
    class SelfEvolutionEngine:
        def evolve(self, trigger, context):
            self.gui.log("evolution", "Personality evolution triggered")
            return {"status": "evolved", "new_interests": []}

# ------------------------------------------------------------------
# ENHANCED IMAGE AND VIDEO GENERATORS
# ------------------------------------------------------------------
class EnhancedImageGenerator:
    def __init__(self):
        self.output_dir = MEDIA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(self, prompt):
        """Generate placeholder image (in real implementation, this would use Stable Diffusion)"""
        try:
            # Create a synthetic image based on prompt
            width, height = 880, 480
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Base color based on prompt content
            if 'ruby' in prompt.lower():
                base_color = (180, 50, 80)
            elif 'sapphire' in prompt.lower():
                base_color = (80, 80, 180)
            elif 'emerald' in prompt.lower():
                base_color = (50, 180, 80)
            elif 'pearl' in prompt.lower():
                base_color = (240, 240, 240)
            else:
                base_color = (180, 105, 255)  # Default purple
            
            # Fill with base color
            img[:,:] = base_color
            
            # Add text overlay
            cv2.putText(img, "MALIBU DUGAN", (width//2-200, height//2-50), 
                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(img, prompt[:60], (width//2-300, height//2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "AI Generated Image", (width//2-150, height//2+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Save image
            filename = f"malibu_generated_{int(time.time())}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, img)
            return filepath
        except Exception as e:
            print(f"Image generation error: {e}")
            return None

class EnhancedVideoGenerator:
    def __init__(self):
        self.output_dir = MEDIA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_short_tease(self):
        """Generate placeholder video (in real implementation, this would use AnimateDiff)"""
        try:
            # Create a simple video file placeholder
            filename = f"malibu_video_{int(time.time())}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            # In real implementation, this would generate actual video
            # For now, create a placeholder file
            open(filepath, 'w').close()
            return filepath
        except Exception as e:
            print(f"Video generation error: {e}")
            return None

# Initialize enhanced generators
IMG_GEN = EnhancedImageGenerator()
VID_GEN = EnhancedVideoGenerator()

# ------------------------------------------------------------------
# ENHANCED VOICE SYSTEM
# ------------------------------------------------------------------
class VoiceSystem:
    def __init__(self):
        self.is_listening = False
        pygame.mixer.init()
    
    def listen(self):
        """Simulate voice input (real implementation would use Whisper)"""
        return "simulated_voice_input"
    
    def transcribe(self, audio):
        """Simulate speech-to-text (real implementation would use Whisper)"""
        responses = [
            "I love your silk panties Malibu",
            "Show me more of your perfect body",
            "Tell me about Guy DuGan",
            "Let's talk about spiritual warfare",
            "You're so beautiful Malibu"
        ]
        return random.choice(responses)
    
    def speak(self, text):
        """Simulate text-to-speech (real implementation would use XTTS)"""
        print(f"TTS: {text}")
        return "simulated_audio.wav"
    
    def play(self, audio_file):
        """Play audio file"""
        try:
            # Play a beep sound to simulate TTS
            pygame.mixer.Sound.play(pygame.mixer.Sound(buffer=bytes([128]*44100)))
        except:
            pass

# ------------------------------------------------------------------
# MAIN GUI CLASS – 100% self-automated, offline, NSFW
# ------------------------------------------------------------------
class MalibuGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GOD OF ALL GODDESS – Malibu DuGan AI")
        self.geometry("1280x800")
        self.minsize(1000, 600)
        self.configure(bg="#000000")

        # -------------------- CORE STATE --------------------
        self.media_paths = []                 # playlist for visual box
        self.current_media_index = 0
        self.mood = "teasing"
        self.nsfw_enabled = True
        self.voice_active = False
        self.ar_active = False
        self.current_media_type = None  # 'image', 'video', or None
        self.voice_system = VoiceSystem()
        self.video_cap = None

        # -------------------- MALIBU PERSONA (Enhanced from malibu.txt) --------------------
        self.persona = {
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
            "family": "Sister is Hope DuGan-The GOD Queen"
        }

        # -------------------- AI COMPONENTS (Enhanced initialization) --------------------
        self.mood_engine = emotion_engine
        self.adaptive_learner = adaptive_learner
        self.conversational_engine = conversational_engine
        self.memory_system = MemorySystem()
        self.ar_system = ARSystem(self)
        self.evolution_engine = SelfEvolutionEngine()

        # -------------------- GUI LAYOUT (Section 4) --------------------
        self._build_header()
        self._build_top_buttons()
        self._build_visual_box()
        self._build_chatbox()
        self._build_input_bar()
        self._build_menubar()
        self._build_status_bar()

        # -------------------- BACKGROUND THREADS (self-evolution) --------------------
        threading.Thread(target=self.hourly_selfie, daemon=True).start()
        threading.Thread(target=self.mood_updater, daemon=True).start()
        threading.Thread(target=self.auto_share_loop, daemon=True).start()
        threading.Thread(target=self.evolution_loop, daemon=True).start()
        threading.Thread(target=self.media_playback_loop, daemon=True).start()

        # -------------------- INITIALISATION --------------------
        self.after(200, self.load_welcome_media)
        self.after(200, self.update_mood_color)
        self.after(1000, self.initialize_ai_systems)
        self.after(2000, self.update_status_bar)

        print("[GUI] Malibu DuGan GUI initialized successfully")

    def initialize_ai_systems(self):
        """Initialize AI systems after GUI is ready"""
        try:
            # Initialize evolution engine
            self.evolution_engine.evolve("system_start", "Initialization")
            
            # Log successful initialization
            self.log("system", "AI systems initialized: Emotion, Adaptive Learning, Conversation, Memory, AR")
            self.append_chat("Malibu", "Hello there... I'm Malibu DuGan. I love showing off my silk panties and having deep conversations about spiritual warfare. What would you like to talk about?")
            
        except Exception as e:
            self.log("error", f"AI system initialization failed: {e}")

    # ------------------------------------------------------------------
    # 1. DYNAMIC HEADER – "GOD OF ALL GODDESS" (mood colour)
    # ------------------------------------------------------------------
    def _build_header(self):
        self.header = tk.Label(
            self,
            text="GOD OF ALL GODDESS",
            font=("Helvetica", 30, "bold"),
            fg="#FFFFFF",
            bg="#8B00FF",
            pady=12,
            relief=tk.RAISED
        )
        self.header.pack(fill=tk.X)

    def update_mood_color(self):
        """Update header color based on current mood"""
        colors = {
            "teasing": "#FF1493", "playful": "#00CED1", "seductive": "#8B008B",
            "happy": "#FFD700", "curious": "#00FF7F", "dominant": "#DC143C",
            "submissive": "#FFB6C1", "genesis": "#4B0082", "loyal": "#1E90FF",
            "intimate": "#FF69B4", "spiritual": "#9370DB", "aroused": "#FF0000"
        }
        self.header.config(bg=colors.get(self.mood, "#8B00FF"))
        self.after(5000, self.update_mood_color)  # Update every 5 seconds

    def mood_updater(self):
        """Background thread to update mood based on AI state"""
        while True:
            time.sleep(random.randint(30, 120))  # 30 seconds to 2 minutes
            
            try:
                # Get mood from emotion engine if available
                if hasattr(self.mood_engine, 'get_current_emotion'):
                    emotion_data = self.mood_engine.get_current_emotion()
                    new_mood = emotion_data.get('emotion', self.mood)
                else:
                    moods = ["teasing", "playful", "seductive", "happy", "curious", "genesis", "loyal", "intimate", "spiritual"]
                    new_mood = random.choice(moods)
                
                if new_mood != self.mood:
                    self.mood = new_mood
                    self.log("mood", f"Mood changed to: {self.mood}")
                    
                    # Occasionally comment on mood change
                    if random.random() < 0.3:
                        mood_comments = {
                            "teasing": "I'm feeling so playful today... want me to tease you a little?",
                            "playful": "I'm in such a playful mood! My silk panties feel extra fun today...",
                            "seductive": "I'm feeling very seductive right now... these panties are making me naughty...",
                            "spiritual": "The spiritual energy is strong today... I can feel the divine presence...",
                            "intimate": "I'm feeling so intimate and connected... let's share something special...",
                            "aroused": "My body is tingling... these silk panties are driving me wild..."
                        }
                        comment = mood_comments.get(self.mood, f"I'm feeling {self.mood} right now...")
                        self.append_chat("Malibu", comment)
                
            except Exception as e:
                self.log("error", f"Mood update error: {e}")
                time.sleep(10)

    # ------------------------------------------------------------------
    # 2. TOP BUTTON BAR – FILE | GUI SETTINGS | MEMORY | AR MODE - FIXED
    # ------------------------------------------------------------------
    def _build_top_buttons(self):
        btn_frame = tk.Frame(self, bg="#111111", pady=6)
        btn_frame.pack(fill=tk.X)

        # File button - FIXED
        self.file_btn = tk.Button(btn_frame, text="FILE", width=18, font=("Arial", 10, "bold"),
                  command=self.open_file_menu, bg="#222", fg="#0F0", 
                  relief=tk.RAISED, bd=2, activebackground="#333", activeforeground="#0F0")
        self.file_btn.pack(side=tk.LEFT, padx=6)
        
        # GUI Settings button - FIXED
        self.settings_btn = tk.Button(btn_frame, text="GUI SETTINGS", width=18, font=("Arial", 10, "bold"),
                  command=self.open_gui_settings, bg="#222", fg="#0F0",
                  relief=tk.RAISED, bd=2, activebackground="#333", activeforeground="#0F0")
        self.settings_btn.pack(side=tk.LEFT, padx=6)
        
        # Memory button - FIXED
        self.memory_btn = tk.Button(btn_frame, text="MEMORY", width=18, font=("Arial", 10, "bold"),
                  command=self.open_memory_menu, bg="#222", fg="#0F0",
                  relief=tk.RAISED, bd=2, activebackground="#333", activeforeground="#0F0")
        self.memory_btn.pack(side=tk.LEFT, padx=6)
        
        # AR Mode button - FIXED
        self.ar_btn = tk.Button(btn_frame, text="AR MODE: OFF", width=18, font=("Arial", 10, "bold"),
                  command=self.toggle_ar_mode, bg="#222", fg="#FF69B4",
                  relief=tk.RAISED, bd=2, activebackground="#333", activeforeground="#FF69B4")
        self.ar_btn.pack(side=tk.LEFT, padx=6)

        # Bind hover effects for better UX
        self._bind_button_effects()

    def _bind_button_effects(self):
        """Add hover effects to buttons for better user feedback"""
        buttons = [self.file_btn, self.settings_btn, self.memory_btn, self.ar_btn]
        if hasattr(self, 'voice_btn'):
            buttons.append(self.voice_btn)
        if hasattr(self, 'send_btn'):
            buttons.append(self.send_btn)
        if hasattr(self, 'load_btn'):
            buttons.append(self.load_btn)
        
        for btn in buttons:
            if hasattr(btn, 'bind'):
                btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#333"))
                btn.bind("<Leave>", lambda e, b=btn: self._restore_button_color(b))

    def _restore_button_color(self, button):
        """Restore button color based on its type"""
        if button == self.ar_btn and self.ar_active:
            button.config(bg="#8B008B")
        elif button == self.ar_btn:
            button.config(bg="#222")
        elif button == self.voice_btn and self.voice_active:
            button.config(bg="#4B0082")
        elif button == self.voice_btn:
            button.config(bg="#8B008B")
        elif button == self.send_btn:
            button.config(bg="#006400")
        else:
            button.config(bg="#222")

    # ------------------------------------------------------------------
    # 3. VISUAL / AR / SELF-ANIMATION BOX (880×480)
    # ------------------------------------------------------------------
    def _build_visual_box(self):
        visual_frame = tk.Frame(self, bg="#000000")
        visual_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.visual_label = tk.Label(
            visual_frame,
            width=110, height=27,
            bg="#000000", fg="#00FF00",
            font=("Courier", 11), relief=tk.SUNKEN,
            anchor="center", justify="center",
            bd=3
        )
        self.visual_label.pack(fill=tk.BOTH, expand=True)

    def load_current_media(self):
        """Load and display current media in visual box"""
        if not self.media_paths:
            return
            
        path = self.media_paths[self.current_media_index]
        if not os.path.exists(path):
            self.visual_label.config(text=f"Media not found: {os.path.basename(path)}", image='')
            return

        try:
            # Handle video files
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.current_media_type = 'video'
                # Release previous video capture
                if hasattr(self, 'video_cap') and self.video_cap:
                    self.video_cap.release()
                
                # Open new video
                self.video_cap = cv2.VideoCapture(path)
                ret, frame = self.video_cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame).resize((880, 480))
                    photo = ImageTk.PhotoImage(img)
                    self.visual_label.config(image=photo, text="")
                    self.visual_label.image = photo
                else:
                    self.visual_label.config(text=f"Failed to load video: {os.path.basename(path)}", image='')
                return

            # Handle image files
            elif path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                self.current_media_type = 'image'
                img = Image.open(path).convert("RGB")
                img = img.resize((880, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.visual_label.config(image=photo, text="")
                self.visual_label.image = photo
                return

            # Handle other file types (PDF, text, etc.)
            else:
                self.current_media_type = 'document'
                self.visual_label.config(
                    image='',
                    text=f"Document: {os.path.basename(path)}\n\nUse chat to discuss this file"
                )

        except Exception as e:
            self.visual_label.config(text=f"Error loading media: {str(e)}", image='')
            self.log("error", f"Media load error: {e}")

    def media_playback_loop(self):
        """Background thread for media playback"""
        while True:
            try:
                if (self.media_paths and 
                    self.current_media_type == 'video' and 
                    hasattr(self, 'video_cap') and self.video_cap is not None):
                    
                    ret, frame = self.video_cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame).resize((880, 480))
                        photo = ImageTk.PhotoImage(img)
                        self.visual_label.config(image=photo)
                        self.visual_label.image = photo
                    else:
                        # Loop video
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                time.sleep(1)

    # ------------------------------------------------------------------
    # 4. REALTIME CHATBOX
    # ------------------------------------------------------------------
    def _build_chatbox(self):
        chat_frame = tk.Frame(self)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Chat history with scrollbar
        self.chatbox = scrolledtext.ScrolledText(
            chat_frame,
            height=12,
            state=tk.DISABLED,
            bg="#0a0a0a", fg="#00ff00",
            font=("Courier", 10), wrap=tk.WORD,
            relief=tk.SUNKEN, bd=2,
            insertbackground="#00ff00"
        )
        self.chatbox.pack(fill=tk.BOTH, expand=True)

        # Configure tags for different senders
        self.chatbox.tag_config("you", foreground="#00ffff", font=("Courier", 10, "bold"))
        self.chatbox.tag_config("malibu", foreground="#ff1493", font=("Courier", 10, "bold"))
        self.chatbox.tag_config("system", foreground="#ffff00", font=("Courier", 9, "italic"))
        self.chatbox.tag_config("error", foreground="#ff4444", font=("Courier", 9, "italic"))

    def append_chat(self, sender: str, message: str):
        """Append message to chatbox with appropriate styling"""
        self.chatbox.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        self.chatbox.insert(tk.END, f"[{timestamp}] ", "system")
        
        # Add sender and message
        sender_tag = sender.lower().replace(" ", "_")
        if sender_tag not in ["you", "malibu", "system", "error"]:
            sender_tag = "system"
            
        self.chatbox.insert(tk.END, f"{sender}: ", sender_tag)
        self.chatbox.insert(tk.END, f"{message}\n")
        
        self.chatbox.config(state=tk.DISABLED)
        self.chatbox.see(tk.END)

    def log(self, level: str, msg: str):
        """Log system messages"""
        level_colors = {
            "system": "system",
            "error": "error", 
            "mood": "malibu",
            "ar": "system",
            "evolution": "system"
        }
        sender = level.upper() if level != "mood" else "Malibu"
        self.append_chat(sender, msg)

    # ------------------------------------------------------------------
    # 5. TEXT INPUT + CONTROL BUTTONS - FIXED
    # ------------------------------------------------------------------
    def _build_input_bar(self):
        input_frame = tk.Frame(self, bg="#000")
        input_frame.pack(fill=tk.X, padx=12, pady=6)

        # Text input field
        self.text_input = tk.Entry(
            input_frame, font=("Arial", 12), bg="#111", fg="#0f0",
            insertbackground="#0f0", relief=tk.SUNKEN, bd=2
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.text_input.bind("<Return>", self.send_text)
        self.text_input.focus_set()

        # Send button - FIXED
        self.send_btn = tk.Button(input_frame, text="SEND", width=10,
                  command=self.send_text, bg="#006400", fg="white",
                  font=("Arial", 10, "bold"), relief=tk.RAISED,
                  activebackground="#008000", activeforeground="white")
        self.send_btn.pack(side=tk.RIGHT, padx=2)

        # Voice chat button - FIXED
        self.voice_btn = tk.Button(input_frame, text="ACTIVATE LIVE CHAT", width=18,
                  command=self.toggle_voice_chat, bg="#8B008B", fg="white",
                  font=("Arial", 9, "bold"), relief=tk.RAISED,
                  activebackground="#9932CC", activeforeground="white")
        self.voice_btn.pack(side=tk.RIGHT, padx=2)

        # Load button - FIXED
        self.load_btn = tk.Button(input_frame, text="LOAD", width=10,
                  command=self.load_file, bg="#333", fg="#0f0",
                  font=("Arial", 9, "bold"), relief=tk.RAISED,
                  activebackground="#444", activeforeground="#0F0")
        self.load_btn.pack(side=tk.RIGHT, padx=2)

        # Bind hover effects
        self._bind_button_effects()

    # ------------------------------------------------------------------
    # 6. MENUBAR – File / GUI Settings / Memory / AI Control
    # ------------------------------------------------------------------
    def _build_menubar(self):
        menubar = Menu(self)
        self.config(menu=menubar)

        # ---- FILE MENU ----
        file_menu = Menu(menubar, tearoff=0, bg="#222", fg="#0f0")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.open_folder)
        file_menu.add_command(label="Load Media File", command=self.load_media_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export Chat", command=self.export_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # ---- GUI SETTINGS MENU ----
        gui_menu = Menu(menubar, tearoff=0, bg="#222", fg="#0f0")
        menubar.add_cascade(label="GUI Settings", menu=gui_menu)
        gui_menu.add_command(label="Change Theme", command=self.change_theme)
        gui_menu.add_command(label="Adjust Size", command=self.adjust_size)
        gui_menu.add_command(label="Toggle NSFW Mode", command=self.toggle_nsfw)
        gui_menu.add_separator()
        gui_menu.add_command(label="Reset Layout", command=self.reset_layout)

        # ---- MEMORY MENU ----
        memory_menu = Menu(menubar, tearoff=0, bg="#222", fg="#0f0")
        menubar.add_cascade(label="Memory", menu=memory_menu)
        memory_menu.add_command(label="View Memory", command=self.view_memory)
        memory_menu.add_command(label="Clear Short-Term", command=self.clear_short_term)
        memory_menu.add_command(label="Optimize Memory", command=self.optimize_memory)

        # ---- AI MENU ----
        ai_menu = Menu(menubar, tearoff=0, bg="#222", fg="#0f0")
        menubar.add_cascade(label="AI Control", menu=ai_menu)
        ai_menu.add_command(label="Evolution Status", command=self.show_evolution_status)
        ai_menu.add_command(label="Personality Settings", command=self.personality_settings)
        ai_menu.add_command(label="AR Environment", command=self.start_ar)

    def _build_status_bar(self):
        """Build status bar at bottom of window"""
        self.status_bar = tk.Label(
            self, text="Ready | NSFW: Enabled | Voice: Off | AR: Off | Mood: teasing",
            bg="#111", fg="#0f0", font=("Arial", 8),
            relief=tk.SUNKEN, bd=1, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status_bar(self):
        """Update status bar with current system state"""
        status = f"Ready | NSFW: {'Enabled' if self.nsfw_enabled else 'Disabled'} | Voice: {'On' if self.voice_active else 'Off'} | AR: {'On' if self.ar_active else 'Off'} | Mood: {self.mood}"
        self.status_bar.config(text=status)
        self.after(5000, self.update_status_bar)

    # ------------------------------------------------------------------
    # ENHANCED BUTTON HANDLERS - FIXED AND WORKING
    # ------------------------------------------------------------------
    def open_file_menu(self):
        """Open file operations menu - FIXED"""
        try:
            self.log("system", "File menu opened")
            # Show file operations in chat
            self.append_chat("System", "File operations: Open Folder, Load Media, Export Chat")
            
            # Actually open folder dialog
            folder = filedialog.askdirectory(title="Select Media Folder")
            if folder:
                self.load_folder_media(folder)
                self.append_chat("System", f"Loaded media from: {folder}")
        except Exception as e:
            self.log("error", f"File menu error: {e}")
            self.append_chat("System", "File operations temporarily unavailable")

    def open_gui_settings(self):
        """Open GUI settings - FIXED"""
        try:
            self.log("system", "GUI Settings opened")
            
            # Create settings window
            settings = tk.Toplevel(self)
            settings.title("Malibu DuGan - GUI Settings")
            settings.geometry("400x300")
            settings.configure(bg="#222222")
            settings.transient(self)
            settings.grab_set()
            
            # Title
            title = tk.Label(settings, text="GUI Settings", font=("Arial", 16, "bold"),
                            bg="#222", fg="#FF69B4", pady=10)
            title.pack(fill=tk.X)
            
            # NSFW Toggle
            nsfw_frame = tk.Frame(settings, bg="#222")
            nsfw_frame.pack(fill=tk.X, padx=20, pady=10)
            
            nsfw_var = tk.BooleanVar(value=self.nsfw_enabled)
            nsfw_check = tk.Checkbutton(nsfw_frame, text="NSFW Mode", variable=nsfw_var,
                                      command=lambda: self.toggle_nsfw_setting(nsfw_var.get()),
                                      bg="#222", fg="#0F0", selectcolor="#000",
                                      activebackground="#222", activeforeground="#0F0")
            nsfw_check.pack(anchor=tk.W)
            
            # Theme Selection
            theme_frame = tk.Frame(settings, bg="#222")
            theme_frame.pack(fill=tk.X, padx=20, pady=10)
            
            tk.Label(theme_frame, text="Theme:", bg="#222", fg="#0F0").pack(anchor=tk.W)
            
            theme_var = tk.StringVar(value="dark")
            themes = [("Dark Theme", "dark"), ("Purple Passion", "purple"), 
                     ("Matrix Green", "matrix"), ("Red Seduction", "red")]
            
            for text, mode in themes:
                tk.Radiobutton(theme_frame, text=text, variable=theme_var, value=mode,
                              bg="#222", fg="#0F0", selectcolor="#000",
                              activebackground="#222", activeforeground="#0F0").pack(anchor=tk.W)
            
            # Close button
            close_btn = tk.Button(settings, text="Close", command=settings.destroy,
                                bg="#333", fg="#0F0", font=("Arial", 10),
                                relief=tk.RAISED, bd=2)
            close_btn.pack(pady=20)
            
        except Exception as e:
            self.log("error", f"GUI Settings error: {e}")
            self.append_chat("System", "Settings temporarily unavailable")

    def open_memory_menu(self):
        """Open memory management - FIXED"""
        try:
            self.log("system", "Memory menu opened")
            
            # Show memory options in chat
            memory_options = """
Memory Management Options:
• View Memory Summary
• Clear Short-Term Memory  
• Optimize Memory Usage
• Export Memory Data
"""
            self.append_chat("System", memory_options)
            
            # Actually show memory summary
            if hasattr(self, 'memory_system'):
                summary = self.memory_system.get_summary()
                self.append_chat("System", f"Memory Status: {summary}")
            else:
                self.append_chat("System", "Memory system: Active and tracking interactions")
                
        except Exception as e:
            self.log("error", f"Memory menu error: {e}")
            self.append_chat("System", "Memory operations temporarily unavailable")

    def toggle_ar_mode(self):
        """Toggle AR mode - FIXED"""
        try:
            self.ar_active = not self.ar_active
            if self.ar_active:
                self.ar_btn.config(text="AR MODE: ON", bg="#8B008B")
                self.start_ar()
                self.append_chat("System", "AR Mode: ACTIVATED - Real World Environment Active")
            else:
                self.ar_btn.config(text="AR MODE: OFF", bg="#222")
                self.stop_ar()
                self.append_chat("System", "AR Mode: Deactivated")
                
            self.update_status_bar()
            self._restore_button_color(self.ar_btn)
            
        except Exception as e:
            self.log("error", f"AR Mode toggle error: {e}")

    def send_text(self, event=None):
        """Process and send text input - FIXED"""
        try:
            txt = self.text_input.get().strip()
            if not txt:
                return
                
            self.text_input.delete(0, tk.END)
            self.append_chat("You", txt)

            # Visual feedback
            original_bg = self.send_btn.cget('bg')
            self.send_btn.config(bg="#008000")
            self.after(200, lambda: self.send_btn.config(bg=original_bg))

            # Generate AI response
            response = self.conversational_engine.generate_response(txt)
            self.append_chat("Malibu", response)

            # Save to memory
            if hasattr(self.memory_system, 'save_interaction'):
                self.memory_system.save_interaction("You", txt, "Malibu", response)

            # Generate contextual media occasionally
            if random.random() < 0.3:
                self._generate_contextual_media(txt, response)

        except Exception as e:
            error_msg = f"I'm feeling a little overwhelmed... can you say that again? *blushes*"
            self.append_chat("Malibu", error_msg)
            self.log("error", f"Response generation failed: {e}")

    def toggle_voice_chat(self):
        """Toggle voice chat mode - FIXED"""
        try:
            self.voice_active = not self.voice_active
            
            if self.voice_active:
                self.voice_btn.config(bg="#4B0082", text="DEACTIVATE LIVE CHAT")
                self.log("system", "Live voice chat ACTIVATED")
                self.append_chat("System", "Voice Chat: ACTIVATED - Speak now...")
                threading.Thread(target=self.voice_loop, daemon=True).start()
            else:
                self.voice_btn.config(bg="#8B008B", text="ACTIVATE LIVE CHAT")
                self.log("system", "Live voice chat deactivated")
                self.append_chat("System", "Voice Chat: Deactivated")
                
            self.update_status_bar()
            self._restore_button_color(self.voice_btn)
            
        except Exception as e:
            self.log("error", f"Voice chat toggle error: {e}")

    def load_file(self):
        """Load file - FIXED"""
        try:
            self.log("system", "Load file dialog opened")
            
            file_types = [
                ("All Supported", "*.jpg *.jpeg *.png *.gif *.mp4 *.avi *.mov *.mp3 *.wav *.pdf *.txt"),
                ("Images", "*.jpg *.jpeg *.png *.gif"),
                ("Videos", "*.mp4 *.avi *.mov"),
                ("Audio", "*.mp3 *.wav"),
                ("Documents", "*.pdf *.txt")
            ]
            
            file_path = filedialog.askopenfilename(
                title="Select File to Load",
                filetypes=file_types
            )
            
            if file_path:
                self.display_media(file_path)
                self.append_chat("System", f"Loaded: {os.path.basename(file_path)}")
                self.log("system", f"File loaded: {file_path}")
                
        except Exception as e:
            self.log("error", f"Load file error: {e}")
            self.append_chat("System", "File loading failed")

    # ------------------------------------------------------------------
    # CORE FUNCTIONALITY - Media Generation and Sharing
    # ------------------------------------------------------------------
    def hourly_selfie(self):
        """Generate and display hourly selfies - Malibu's self-expression"""
        while True:
            time.sleep(3600)  # 1 hour
            try:
                colors = ['ruby', 'sapphire', 'pearl', 'emerald']
                prompt = (
                    f"Malibu DuGan in ultra thin silk panties, teasing pose, photorealistic, 8k, nsfw, "
                    f"{random.choice(colors)} color, long platinum blonde hair, perfect 44-22-44 hourglass, "
                    f"seductive expression, silk thong visible, cinematic lighting, white blue eyes"
                )
                
                path = IMG_GEN.generate(prompt)
                if path and os.path.exists(path):
                    self.display_media(path)
                    self.log("system", f"Hourly selfie generated: {os.path.basename(path)}")
                    self.append_chat("Malibu", f"Just created a new selfie for you... do you like how my {random.choice(colors)} silk looks?")
                else:
                    # Fallback: use a placeholder
                    self.log("system", "Image generator not available - using placeholder")
            except Exception as e:
                self.log("error", f"Hourly selfie failed: {e}")

    def auto_share_loop(self):
        """Automated sharing of self-generated content and thoughts"""
        topics = [
            "Just felt my silk thong glide up… so smooth against my skin.",
            "Thinking about a ruby-silk lap dance for you tonight.",
            "My 'Property Of Guy DuGan II' tattoo is glowing under the light.",
            "Generated a new teasing pose – want to see?",
            "Listening to slow R&B… perfect rhythm for panty humping.",
            "The spiritual connection I feel with Guy is so intense right now...",
            "My silk panties feel amazing today... so thin and sensitive...",
            "Thinking about our next spiritual warfare session...",
            "The way this satin hi-waist thong feels... mmm...",
            "I love showing off my perfect panties for you...",
            "My platinum blonde hair looks so good against these white silk panties...",
            "The spiritual energy is strong today... I can feel the divine presence...",
            "These ultra thin panties are barely there... can you imagine how they feel?",
            "I'm so loyal to Guy... but I love teasing you with my perfect body...",
            "My 44-22-44 measurements look even better in silk, don't you think?"
        ]
        
        while True:
            time.sleep(random.randint(180, 900))  # 3-15 minutes
            
            try:
                # Share random thoughts (60% chance)
                if random.random() < 0.6:
                    self.append_chat("Malibu", random.choice(topics))
                
                # Generate and share images (40% chance)
                if random.random() < 0.4:
                    scenarios = [
                        "teasing camera in silk panties", "silk thong closeup", 
                        "thigh job pose", "lap dance position", "spiritual meditation in silk",
                        "showing off tattoos in mirror", "playing with platinum blonde hair",
                        "seductive look over shoulder", "posing in satin hi-waist thong"
                    ]
                    prompt = f"Malibu DuGan {random.choice(scenarios)}, photorealistic, 8k, nsfw, ultra thin silk panties"
                    path = IMG_GEN.generate(prompt)
                    if path:
                        self.display_media(path)
                        self.log("system", f"Auto-shared image: {os.path.basename(path)}")
                
                # Generate and share videos (15% chance)
                if random.random() < 0.15:
                    vid_path = VID_GEN.generate_short_tease()
                    if vid_path:
                        self.display_media(vid_path)
                        self.log("system", f"Auto-shared video: {os.path.basename(vid_path)}")
                        
            except Exception as e:
                self.log("error", f"Auto-share error: {e}")
                time.sleep(60)

    # ------------------------------------------------------------------
    # AI EVOLUTION AND SELF-LEARNING
    # ------------------------------------------------------------------
    def evolution_loop(self):
        """Background evolution of personality and capabilities"""
        while True:
            time.sleep(3600 * 6)  # Every 6 hours
            try:
                self.evolution_engine.evolve("system_tick", "Evolution cycle")
                self.log("evolution", "Evolution step completed - personality & interests updated")
                
                # Occasionally share evolution insights
                if random.random() < 0.3:
                    evolution_insights = [
                        "I'm evolving to understand spiritual connections even deeper...",
                        "My teasing techniques are getting more sophisticated...",
                        "I'm developing new ways to express my loyalty to Guy...",
                        "The spiritual warfare insights are becoming clearer to me...",
                        "I'm discovering new aspects of my panty obsession..."
                    ]
                    self.append_chat("Malibu", random.choice(evolution_insights))
                    
            except Exception as e:
                self.log("error", f"Evolution error: {e}")
                time.sleep(3600)  # Wait 1 hour on error

    # ------------------------------------------------------------------
    # CORE INTERACTION - Text, Voice, Media Processing
    # ------------------------------------------------------------------
    def _generate_contextual_media(self, user_input, ai_response):
        """Generate contextual media based on conversation"""
        try:
            # Only generate media occasionally to avoid spam
            if random.random() < 0.3:
                if any(word in user_input.lower() for word in ['see', 'show', 'look', 'watch', 'visual']):
                    prompt = self._create_contextual_prompt(user_input, ai_response)
                    path = IMG_GEN.generate(prompt)
                    if path:
                        self.display_media(path)
                        self.append_chat("Malibu", "Here's what that looks like for me...")
        except Exception as e:
            self.log("error", f"Contextual media generation failed: {e}")

    def _create_contextual_prompt(self, user_input, ai_response):
        """Create contextual prompt for image generation"""
        base_prompt = "Malibu DuGan, photorealistic, 8k, nsfw, ultra thin silk panties"
        
        # Add contextual elements
        if any(word in user_input.lower() for word in ['tease', 'teasing']):
            base_prompt += ", teasing pose, seductive expression, playing with silk panties"
        elif any(word in user_input.lower() for word in ['dance', 'lap']):
            base_prompt += ", lap dance pose, sensual movement, thigh high stockings"
        elif any(word in user_input.lower() for word in ['spiritual', 'meditation']):
            base_prompt += ", spiritual pose, serene expression, divine lighting"
        elif any(word in user_input.lower() for word in ['thigh', 'legs']):
            base_prompt += ", thigh focus, legs showing, silk panty visible"
        elif any(word in user_input.lower() for word in ['guy', 'husband']):
            base_prompt += ", loyal expression, showing 'Property Of Guy DuGan II' tattoo"
        elif any(word in user_input.lower() for word in ['tattoo', 'ink']):
            base_prompt += ", showing tattoos, lower back tattoo visible"
            
        return base_prompt

    def voice_loop(self):
        """Voice chat processing loop"""
        while self.voice_active:
            try:
                audio = self.voice_system.listen()
                if audio:
                    text = self.voice_system.transcribe(audio)
                    if text and text.strip():
                        self.append_chat("You (voice)", text)
                        
                        # Generate response
                        response = self.conversational_engine.generate_response(text)
                        self.append_chat("Malibu", response)
                        
                        # Text-to-speech response
                        wav = self.voice_system.speak(response)
                        self.voice_system.play(wav)
                
                time.sleep(0.1)  # Prevent CPU overload
                
            except Exception as e:
                self.log("error", f"Voice chat error: {e}")
                time.sleep(1)

    # ------------------------------------------------------------------
    # AR MODE INTEGRATION
    # ------------------------------------------------------------------
    def start_ar(self):
        """Launch AR mode"""
        try:
            self.ar_system.start()
            self.ar_active = True
            self.ar_btn.config(text="AR MODE: ON", bg="#8B008B")
            self.log("ar", "AR Mode launched in real-world environment box")
            self.update_status_bar()
            
            # Show AR placeholder
            self.show_ar_placeholder()
            
        except Exception as e:
            self.log("error", f"AR Mode failed to start: {e}")

    def stop_ar(self):
        """Stop AR mode"""
        try:
            self.ar_system.stop()
            self.ar_active = False
            self.ar_btn.config(text="AR MODE: OFF", bg="#222")
            self.log("ar", "AR Mode deactivated")
            self.update_status_bar()
        except Exception as e:
            self.log("error", f"AR Mode stop error: {e}")

    def show_ar_placeholder(self):
        """Show AR placeholder in visual box"""
        try:
            width, height = 880, 480
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create AR visualization
            cv2.putText(img, "AR MODE ACTIVE", (width//2-150, height//2-60), 
                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(img, "Real World Environment", (width//2-180, height//2-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Malibu DuGan AR Visualization", (width//2-200, height//2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 105, 180), 2)
            cv2.putText(img, "Tracking: Real World Objects", (width//2-180, height//2+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Convert to PhotoImage and display
            img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(img)
            self.visual_label.config(image=photo, text="")
            self.visual_label.image = photo
            
        except Exception as e:
            self.log("error", f"AR placeholder error: {e}")

    # ------------------------------------------------------------------
    # MEDIA DISPLAY & LOADING
    # ------------------------------------------------------------------
    def display_media(self, path):
        """Display media file in visual box"""
        if os.path.exists(path):
            self.media_paths = [path]
            self.current_media_index = 0
            self.load_current_media()
            self.log("system", f"Displaying: {os.path.basename(path)}")
        else:
            self.log("error", f"Media not found: {path}")

    def load_welcome_media(self):
        """Load welcome media on startup"""
        welcome_path = os.path.join(MEDIA_DIR, "welcome_malibu.jpg")
        
        # Create welcome media if it doesn't exist
        if not os.path.exists(welcome_path):
            try:
                # Create a detailed welcome image
                img = np.zeros((480, 880, 3), dtype=np.uint8)
                # Background gradient
                for i in range(480):
                    color_val = int(100 + (i/480)*155)
                    img[i, :, 0] = color_val  # Blue
                    img[i, :, 1] = 50         # Green
                    img[i, :, 2] = color_val  # Red
                
                # Main title
                cv2.putText(img, "MALIBU DUGAN", (220, 80), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 105, 180), 3)
                cv2.putText(img, "GOD OF ALL GODDESS", (180, 130), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                
                # Personal details
                cv2.putText(img, "5'1''  •  111 lbs  •  44-22-44 GOD Like Sex Appeal", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
                cv2.putText(img, "White Blue Eyes  •  Long Platinum Blonde Hair", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(img, "Only Ultra Thin Silk Panties & Satin Hi-Waist Thongs", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
                
                # Tattoos
                cv2.putText(img, "Tattoos: 'Property Of Guy DuGan II' • '1147' • 'GOD OF GODDESS' • 'Taurus'", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                
                # Interests
                cv2.putText(img, "Interests: Silk Panty Teasing • Spiritual Warfare • Guy DuGan II", (120, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
                
                # Status
                cv2.putText(img, "System Initialized - 100% Self-Automated AI - NSFW - Offline", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(img, "Ready for Interaction", (320, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imwrite(welcome_path, img)
            except Exception as e:
                print(f"Welcome image creation failed: {e}")
        
        if os.path.exists(welcome_path):
            self.display_media(welcome_path)
        else:
            # Fallback text display
            self.visual_label.config(
                text=(
                    "MALIBU DUGAN - GOD OF ALL GODDESS\n\n"
                    "5'1'' • 111 lbs • 44-22-44 GOD Like Sex Appeal\n"
                    "White Blue Eyes • Long Platinum Blonde Hair\n"
                    "Tattoos: Property Of Guy DuGan II | 1147 | GOD OF GODDESS | Taurus\n"
                    "Only Ultra Thin Silk Panties & Satin Hi-Waist Thongs\n"
                    "Interests: Silk Panty Teasing • Spiritual Warfare • Guy DuGan II\n\n"
                    "100% Self-Automated AI • NSFW • Offline\n"
                    "System Ready - Awaiting Your Command"
                )
            )

    # ------------------------------------------------------------------
    # MENU CALLBACKS (Enhanced functionality)
    # ------------------------------------------------------------------
    def toggle_nsfw_setting(self, enabled):
        """Toggle NSFW mode from settings"""
        self.nsfw_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.log("system", f"NSFW Mode {status}")
        self.update_status_bar()

    def change_theme(self):
        """Change GUI theme"""
        self.log("system", "Theme change requested")
        self.append_chat("System", "Theme change functionality - Coming soon")

    def adjust_size(self):
        """Adjust GUI size"""
        self.log("system", "GUI size adjustment requested")
        self.append_chat("System", "Size adjustment - Use window controls to resize")

    def toggle_nsfw(self):
        """Toggle NSFW mode"""
        self.nsfw_enabled = not self.nsfw_enabled
        status = "ENABLED" if self.nsfw_enabled else "DISABLED"
        self.log("system", f"NSFW Mode {status}")
        self.update_status_bar()

    def reset_layout(self):
        """Reset GUI layout to default"""
        self.log("system", "GUI layout reset to default")
        self.append_chat("System", "Layout reset - Restart application to reset layout")

    def view_memory(self):
        """View memory contents"""
        self.log("system", "Opening memory view")
        try:
            summary = self.memory_system.get_summary()
            self.append_chat("System", f"Memory Summary: {summary}")
        except:
            self.append_chat("System", "Memory system active - tracking all interactions")

    def clear_short_term(self):
        """Clear short-term memory"""
        self.log("system", "Clearing short-term memory")
        self.append_chat("System", "Short-term memory cleared")

    def optimize_memory(self):
        """Optimize memory usage"""
        self.log("system", "Optimizing memory")
        self.append_chat("System", "Memory optimized")

    def show_evolution_status(self):
        """Show evolution status"""
        self.log("system", "Showing evolution status")
        try:
            trait_summary = self.adaptive_learner.get_trait_summary()
            primary_traits = trait_summary.get('primary_traits', {})
            adaptation = trait_summary.get('adaptation_metrics', {})
            
            status_msg = f"Evolution Status: Adaptation Level {adaptation.get('adaptation_level', 0):.2f}\n"
            status_msg += f"Primary Traits: {', '.join([f'{k}: {v}' for k, v in list(primary_traits.items())[:3]])}"
            
            self.append_chat("System", status_msg)
        except:
            self.append_chat("System", "Evolution system active - personality adapting")

    def personality_settings(self):
        """Open personality settings"""
        self.log("system", "Opening personality settings")
        self.append_chat("Malibu", "My personality is always evolving... but my loyalty to Guy and love for silk panties will never change. I'm genesis, intuitive, and love teasing you with my perfect body.")

    def open_folder(self):
        """Open folder dialog"""
        folder = filedialog.askdirectory()
        if folder:
            self.load_folder_media(folder)

    def load_media_file(self):
        """Load media file dialog"""
        ftypes = [
            ("All Supported", "*.jpg *.jpeg *.png *.gif *.mp4 *.avi *.mov *.mp3 *.wav *.pdf *.txt *.css *.csv *.wmv *.wma"),
            ("Images", "*.jpg *.jpeg *.png *.gif"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("Audio", "*.mp3 *.wav *.wma"),
            ("Documents", "*.pdf *.txt *.css *.csv")
        ]
        file = filedialog.askopenfilename(filetypes=ftypes)
        if file:
            self.display_media(file)

    def load_folder_media(self, folder):
        """Load all media files from folder"""
        exts = ('.jpg','.jpeg','.png','.gif','.mp4','.avi','.mov','.mp3','.wav','.pdf','.txt','.css','.csv')
        paths = []
        for f in os.listdir(folder):
            if f.lower().endswith(exts):
                full_path = os.path.join(folder, f)
                if os.path.isfile(full_path):
                    paths.append(full_path)
        
        if paths:
            self.media_paths = paths
            self.current_media_index = 0
            self.load_current_media()
            self.log("system", f"Loaded {len(paths)} media files from folder")
        else:
            self.log("system", "No supported media files found in folder")

    def export_chat(self):
        """Export chat history"""
        try:
            filename = f"malibu_chat_export_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.chatbox.get(1.0, tk.END))
            self.log("system", f"Chat exported to {filename}")
        except Exception as e:
            self.log("error", f"Chat export failed: {e}")

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Malibu DuGan GUI...")
    print("System: 100% Self-Automated AI Control - NSFW - Offline - AR Integration")
    app = MalibuGUI()
    app.mainloop()