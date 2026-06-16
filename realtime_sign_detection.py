import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import math
import os
import tkinter as tk
from tkinter import Label, Button, Frame, StringVar
from PIL import Image, ImageTk
import threading
import pyttsx3
import time
from collections import deque
import queue

# ✅ Load trained model
model = load_model("sign_language_model.h5")

# ✅ Class labels (auto-fetch class names)
data_dir = r"F:\Sign language detection\dist\Data"
labels = sorted(os.listdir(data_dir))

# ✅ Hand detector setup
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# ✅ Initialize camera
cap = cv2.VideoCapture(0)
running = False

# ✅ Speech system with queue
speech_queue = queue.Queue()
last_spoken_time = 0
speech_cooldown = 3  # seconds between speech
confidence_threshold = 0.60

# ✅ For smoothing predictions
prediction_history = deque(maxlen=5)

# ✅ Initialize speech engine properly
def init_speech_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        return engine
    except Exception as e:
        print(f"Speech engine init error: {e}")
        return None

speech_engine = init_speech_engine()

def speech_worker():
    """Background worker for speech synthesis"""
    while True:
        try:
            text = speech_queue.get()
            if text is None:  # Shutdown signal
                break
                
            if speech_engine:
                speech_engine.say(text)
                speech_engine.runAndWait()
            time.sleep(0.1)
            speech_queue.task_done()
        except Exception as e:
            print(f"Speech error: {e}")
            time.sleep(0.1)

# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_label(text):
    """Alternative speech function using system voice"""
    try:
        import subprocess
        import platform
        import os
        
        # Use system's text-to-speech
        if platform.system() == "Windows":
            # Windows - using PowerShell speech
            subprocess.run([
                "powershell", 
                "-Command", 
                f"Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak('{text}')"
            ], capture_output=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["say", text])
        else:  # Linux
            subprocess.run(["espeak", text])
        return True
    except Exception as e:
        print(f"Alternative speech error: {e}")
        return False

# ==============================
# CONTROL FUNCTIONS
# ==============================
def start_detection():
    global running
    if not running:
        running = True
        status_var.set("🟢 Starting camera...")
        threading.Thread(target=run_camera, daemon=True).start()

def stop_detection():
    global running
    running = False
    status_var.set("🟠 Stopping...")

# ==============================
# MODERN GUI SETUP - FIXED LAYOUT
# ==============================
root = tk.Tk()
root.title("🤟 Sign Language Detection Pro")
root.geometry("1200x800")
root.configure(bg="#0F0F23")

# Custom colors
BG_COLOR = "#0F0F23"
CARD_COLOR = "#1E1E3F"
ACCENT_COLOR = "#00FFAA"
WARNING_COLOR = "#FF4444"
TEXT_COLOR = "#FFFFFF"
SECONDARY_TEXT = "#B0B0B0"

# Main container with proper layout management
main_container = Frame(root, bg=BG_COLOR)
main_container.pack(fill="both", expand=True, padx=20, pady=20)

# Header
header_frame = Frame(main_container, bg=BG_COLOR)
header_frame.pack(fill="x", pady=(0, 20))

title_label = Label(header_frame, text="🤟 SIGN LANGUAGE DETECTOR", 
                   font=("Arial", 24, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)
title_label.pack()

subtitle_label = Label(header_frame, text="Real-time Sign Language Detection with Voice Output", 
                      font=("Arial", 12), bg=BG_COLOR, fg=SECONDARY_TEXT)
subtitle_label.pack(pady=(5, 0))

# Content area with proper weight distribution
content_frame = Frame(main_container, bg=BG_COLOR)
content_frame.pack(fill="both", expand=True)

# Configure grid for proper layout
content_frame.columnconfigure(0, weight=3)  # Video takes 75%
content_frame.columnconfigure(1, weight=1)  # Controls take 25%
content_frame.rowconfigure(0, weight=1)

# Left panel - Video feed (75% width)
left_panel = Frame(content_frame, bg=CARD_COLOR, relief="flat", bd=2)
left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

video_header = Label(left_panel, text="📷 LIVE CAMERA FEED", font=("Arial", 14, "bold"), 
                    bg=CARD_COLOR, fg=TEXT_COLOR, pady=10)
video_header.pack(fill="x")

video_label = Label(left_panel, bg=CARD_COLOR)
video_label.pack(padx=20, pady=20, fill="both", expand=True)

# Right panel - Controls and info (25% width)
right_panel = Frame(content_frame, bg=BG_COLOR)
right_panel.grid(row=0, column=1, sticky="nsew")

# Create a container with fixed height for right panel content
right_content = Frame(right_panel, bg=BG_COLOR)
right_content.pack(fill="both", expand=True)

# Detection card - FIXED HEIGHT
detection_card = Frame(right_content, bg=CARD_COLOR, relief="flat", bd=2, padx=15, pady=15)
detection_card.pack(fill="x", pady=(0, 10))

detection_title = Label(detection_card, text="🎯 DETECTION RESULTS", 
                       font=("Arial", 12, "bold"), bg=CARD_COLOR, fg=TEXT_COLOR)
detection_title.pack(anchor="w", pady=(0, 10))

# Detection result
result_frame = Frame(detection_card, bg=CARD_COLOR)
result_frame.pack(fill="x", pady=5)

Label(result_frame, text="Detected Sign:", font=("Arial", 10), 
      bg=CARD_COLOR, fg=SECONDARY_TEXT).pack(anchor="w")

detected_sign_var = StringVar(value="Waiting for detection...")
detected_sign_label = Label(result_frame, textvariable=detected_sign_var,
                          font=("Arial", 16, "bold"), bg=CARD_COLOR, fg=ACCENT_COLOR)
detected_sign_label.pack(anchor="w", pady=(2, 0))

# Confidence meter
confidence_frame = Frame(detection_card, bg=CARD_COLOR)
confidence_frame.pack(fill="x", pady=8)

Label(confidence_frame, text="Confidence Level:", font=("Arial", 10), 
      bg=CARD_COLOR, fg=SECONDARY_TEXT).pack(anchor="w")

confidence_canvas = tk.Canvas(confidence_frame, bg="#333366", height=20, width=180, highlightthickness=0)
confidence_canvas.pack(fill="x", pady=(3, 0))

confidence_bar_id = confidence_canvas.create_rectangle(0, 0, 0, 20, fill=ACCENT_COLOR, outline="")

confidence_text_var = StringVar(value="0%")
confidence_text = Label(confidence_frame, textvariable=confidence_text_var, font=("Arial", 9), 
                       bg=CARD_COLOR, fg=SECONDARY_TEXT)
confidence_text.pack(anchor="e")

# Speech status
speech_frame = Frame(detection_card, bg=CARD_COLOR)
speech_frame.pack(fill="x", pady=5)

Label(speech_frame, text="Speech Status:", font=("Arial", 10), 
      bg=CARD_COLOR, fg=SECONDARY_TEXT).pack(anchor="w")

speech_status_var = StringVar(value="🔇 Ready")
speech_status_label = Label(speech_frame, textvariable=speech_status_var,
                          font=("Arial", 10, "bold"), bg=CARD_COLOR, fg=ACCENT_COLOR)
speech_status_label.pack(anchor="w", pady=(2, 0))

# Controls card - FIXED HEIGHT
controls_card = Frame(right_content, bg=CARD_COLOR, relief="flat", bd=2, padx=15, pady=15)
controls_card.pack(fill="x", pady=(0, 10))

controls_title = Label(controls_card, text="⚙️ CONTROLS", 
                      font=("Arial", 12, "bold"), bg=CARD_COLOR, fg=TEXT_COLOR)
controls_title.pack(anchor="w", pady=(0, 10))

# Button creation function
def create_modern_button(parent, text, command, color):
    btn = Button(parent, text=text, command=command,
                font=("Arial", 11, "bold"), bg=color, fg="white",
                relief="flat", bd=0, padx=15, pady=10, cursor="hand2")
    btn.pack(fill="x", pady=6)
    return btn

# Create buttons with proper spacing
start_btn = create_modern_button(controls_card, "▶ START DETECTION", start_detection, "#00CC88")
stop_btn = create_modern_button(controls_card, "⏹ STOP DETECTION", stop_detection, WARNING_COLOR)
exit_btn = create_modern_button(controls_card, "❌ EXIT", root.destroy, "#666699")

# Status card - FIXED HEIGHT
status_card = Frame(right_content, bg=CARD_COLOR, relief="flat", bd=2, padx=15, pady=15)
status_card.pack(fill="x")

status_title = Label(status_card, text="📊 SYSTEM STATUS", 
                    font=("Arial", 12, "bold"), bg=CARD_COLOR, fg=TEXT_COLOR)
status_title.pack(anchor="w", pady=(0, 8))

status_var = StringVar(value="🔴 Ready to start")
status_label = Label(status_card, textvariable=status_var, font=("Arial", 10), 
                    bg=CARD_COLOR, fg=SECONDARY_TEXT, wraplength=250, justify="left")
status_label.pack(anchor="w")

# Add some empty space at the bottom to prevent crowding
spacer = Frame(right_content, bg=BG_COLOR, height=20)
spacer.pack(fill="x", side="bottom")

# ==============================
# CAMERA FUNCTIONS
# ==============================
def update_confidence_bar(confidence):
    """Update the confidence bar visualization"""
    width = int(confidence * 180)  # 180 is the width of the canvas
    confidence_canvas.coords(confidence_bar_id, 0, 0, width, 20)
    confidence_text_var.set(f"{confidence*100:.0f}%")
    
    # Change color based on confidence
    if confidence >= 0.8:
        confidence_canvas.itemconfig(confidence_bar_id, fill="#00FFAA")
    elif confidence >= 0.6:
        confidence_canvas.itemconfig(confidence_bar_id, fill="#FFAA00")
    else:
        confidence_canvas.itemconfig(confidence_bar_id, fill=WARNING_COLOR)

def run_camera():
    global running
    
    status_var.set("🟢 Detection running...")
    last_detected_label = None
    
    while running:
        success, img = cap.read()
        if not success:
            continue

        hands, img = detector.findHands(img, draw=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Prepare image for prediction
                img_input = cv2.resize(imgWhite, (300, 300))
                img_input = np.expand_dims(img_input / 255.0, axis=0)

                prediction = model.predict(img_input, verbose=0)
                prob = np.max(prediction)
                index = np.argmax(prediction)
                label = labels[index]

                # Add to prediction history for smoothing
                prediction_history.append((label, prob))
                
                # Get most frequent prediction from history
                if len(prediction_history) >= 3:
                    recent_labels = [pred[0] for pred in prediction_history]
                    most_common = max(set(recent_labels), key=recent_labels.count)
                    if recent_labels.count(most_common) >= 3:
                        label = most_common
                        prob = np.mean([p[1] for p in prediction_history if p[0] == most_common])

                # Update GUI
                if prob >= confidence_threshold:
                    detected_sign_var.set(f"{label.upper()}")
                    update_confidence_bar(prob)
                    
                    # Speech logic
                    if label != last_detected_label:
                        if speak_label(label):
                            speech_status_var.set(f"🔊 Speaking: {label}")
                            root.after(2000, lambda: speech_status_var.set("🔇 Ready"))
                        last_detected_label = label
                        
                else:
                    detected_sign_var.set("UNCERTAIN")
                    update_confidence_bar(prob)
                    last_detected_label = None

                # Draw result on camera feed
                cv2.rectangle(img, (x - offset, y - offset),
                            (x + w + offset, y + h + offset),
                            (0, 255, 0), 3)
                cv2.putText(img, f"{label} ({prob*100:.0f}%)", (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Processing error: {e}")
                detected_sign_var.set("PROCESSING ERROR")
                update_confidence_bar(0)
                last_detected_label = None
        else:
            # No hands detected
            detected_sign_var.set("NO HAND DETECTED")
            update_confidence_bar(0)
            last_detected_label = None
            prediction_history.clear()

        # Convert for Tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        video_label.img_tk = img_tk
        video_label.config(image=img_tk)

    # Cleanup when stopped
    status_var.set("🔴 Detection stopped")
    detected_sign_var.set("Waiting for detection...")
    update_confidence_bar(0)
    prediction_history.clear()

# Cleanup function
def on_closing():
    global running
    running = False
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI
root.mainloop()