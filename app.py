from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import cv2
import mediapipe as mp
import joblib
from PIL import Image
import base64
from io import BytesIO
import os

app = Flask(__name__)
app.secret_key = "asl_secret_key"

# Helper to normalize landmarks (must match training and webcam exactly)
def extract_and_normalize_landmarks(hand_landmarks, width, height):
    # Convert to absolute pixel coordinates to make it aspect-ratio independent
    coords = np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords_relative = coords - wrist
    flattened = coords_relative.flatten()
    norm = np.linalg.norm(flattened)
    if norm > 0:
        flattened = flattened / norm
    return flattened

# Model path
model_path = "asl_rf_model.joblib"

# Load Random Forest model if it exists, otherwise print error
if os.path.exists(model_path):
    print(f"Loading model '{model_path}'...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    print(f"Warning: '{model_path}' not found! App will run, but predictions will fail until you train the model.")
    model = None

# Initialize MediaPipe Hands detectors on startup
mp_hands = mp.solutions.hands
hands_detector_default = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
hands_detector_low = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.10
)

# Helper function to map landmarks back to original image space
def map_landmarks_to_original(hand_landmarks, w_orig, h_orig, pad):
    mapped_list = []
    w_pad = w_orig + 2 * pad
    h_pad = h_orig + 2 * pad
    for lm in hand_landmarks.landmark:
        x_mapped = (lm.x * w_pad - pad) / w_orig if pad > 0 else lm.x
        y_mapped = (lm.y * h_pad - pad) / h_orig if pad > 0 else lm.y
        mapped_list.append({
            "x": float(x_mapped),
            "y": float(y_mapped),
            "z": float(lm.z)
        })
    return mapped_list

# Helper function to generate preprocessed images for fallback detection
def get_preprocessed_variations(img_np):
    pad30_reflect = cv2.copyMakeBorder(img_np, 30, 30, 30, 30, borderType=cv2.BORDER_REFLECT)
    pad30_reflect_gray = cv2.cvtColor(cv2.cvtColor(pad30_reflect, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    
    return [
        ("Raw", img_np, 0),
        ("Pad 30 Reflect + Grayscale", pad30_reflect_gray, 30),
        ("Pad 15 Reflect", cv2.copyMakeBorder(img_np, 15, 15, 15, 15, borderType=cv2.BORDER_REFLECT), 15),
        ("Scale 1.3 b=30", cv2.convertScaleAbs(img_np, alpha=1.3, beta=30), 0),
        ("Pad 30 Constant + Scale a=1.3 b=10", cv2.convertScaleAbs(cv2.copyMakeBorder(img_np, 30, 30, 30, 30, borderType=cv2.BORDER_CONSTANT, value=[0,0,0]), alpha=1.3, beta=10), 30),
        ("Scale 0.7 b=10", cv2.convertScaleAbs(img_np, alpha=0.7, beta=10), 0),
        ("Pad 15 Constant + Scale a=1.3 b=30", cv2.convertScaleAbs(cv2.copyMakeBorder(img_np, 15, 15, 15, 15, borderType=cv2.BORDER_CONSTANT, value=[0,0,0]), alpha=1.3, beta=30), 15)
    ]

@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize session variables
    if "word" not in session:
        session["word"] = ""
    if "last_prediction" not in session:
        session["last_prediction"] = None
    if "last_confidence" not in session:
        session["last_confidence"] = None

    if request.method == 'POST':
        action = request.form.get("action")

        # Clear word
        if action == "clear":
            session["word"] = ""
            session["last_prediction"] = None
            session["last_confidence"] = None
            return redirect(url_for("home"))

        # Add to word using stored prediction
        if action == "add":
            letter = session.get("last_prediction")

            if letter:
                if letter == "SPACE":
                    session["word"] += " "
                elif letter == "DEL":
                    session["word"] = session["word"][:-1]
                elif letter != "NOTHING":
                    session["word"] += letter

            return redirect(url_for("home"))

        # Predict action (Image Upload)
        if action == "predict":
            file = request.files.get('image')

            if file and model is not None:
                try:
                    from PIL import ImageOps
                    img = Image.open(file)
                    img = ImageOps.exif_transpose(img)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    img_np = np.array(img)
                    h_orig, w_orig, _ = img_np.shape
                    
                    # Run highest confidence fallback selector
                    variations = get_preprocessed_variations(img_np)
                    best_letter = "NOTHING"
                    best_confidence = 0.0
                    best_landmarks_list = None
                    
                    for name, prep, pad in variations:
                        detector = hands_detector_default if name == "Raw" else hands_detector_low
                        results = detector.process(prep)
                        if results.multi_hand_landmarks:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            h, w, _ = prep.shape
                            features = extract_and_normalize_landmarks(hand_landmarks, w, h)
                            
                            probabilities = model.predict_proba([features])[0]
                            confidence = float(np.max(probabilities))
                            predicted_class_idx = np.argmax(probabilities)
                            letter = model.classes_[predicted_class_idx]
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_letter = letter
                                best_landmarks_list = map_landmarks_to_original(hand_landmarks, w_orig, h_orig, pad)
                                
                    if best_landmarks_list is not None:
                        session["last_prediction"] = best_letter
                        session["last_confidence"] = best_confidence
                        session["last_landmarks"] = best_landmarks_list
                    else:
                        session["last_prediction"] = "NOTHING"
                        session["last_confidence"] = 0.0
                        session["last_landmarks"] = None
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error predicting uploaded image: {e}")
                    session["last_prediction"] = "ERROR"
                    session["last_confidence"] = 0.0
                    session["last_landmarks"] = None

            return redirect(url_for("home"))

    import json
    landmarks_json = None
    if session.get("last_landmarks") is not None:
        try:
            landmarks_json = json.dumps(session.get("last_landmarks"))
        except Exception:
            landmarks_json = None

    return render_template(
        'index.html',
        prediction=session.get("last_prediction"),
        confidence=session.get("last_confidence"),
        word=session.get("word"),
        landmarks_json=landmarks_json
    )

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded. Train it first."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Ensure image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_np = np.array(img)
        
        # Process image with MediaPipe (Raw first for speed)
        results = hands_detector_default.process(img_np)
        h_orig, w_orig, _ = img_np.shape
        
        selected_hand_landmarks = None
        selected_pad = 0
        
        if results.multi_hand_landmarks:
            selected_hand_landmarks = results.multi_hand_landmarks[0]
            selected_pad = 0
        else:
            # Sequential fallback for webcam to maintain real-time speed while recovering detection
            variations = get_preprocessed_variations(img_np)
            for name, prep, pad in variations:
                if name == "Raw":
                    continue # Already tried Raw
                
                res = hands_detector_low.process(prep)
                if res.multi_hand_landmarks:
                    selected_hand_landmarks = res.multi_hand_landmarks[0]
                    selected_pad = pad
                    break
        
        if selected_hand_landmarks is not None:
            # Extract features using the processed image size
            h_processed = h_orig + 2 * selected_pad
            w_processed = w_orig + 2 * selected_pad
            features = extract_and_normalize_landmarks(selected_hand_landmarks, w_processed, h_processed)
            
            # Run Random Forest prediction
            probabilities = model.predict_proba([features])[0]
            confidence = float(np.max(probabilities))
            predicted_class_idx = np.argmax(probabilities)
            letter = model.classes_[predicted_class_idx]
            
            # Format confidence as percentage
            confidence_percent = f"{confidence * 100:.1f}%"
            
            # Map landmarks back to original coordinate space
            landmarks_list = map_landmarks_to_original(selected_hand_landmarks, w_orig, h_orig, selected_pad)
            
            return jsonify({
                "status": "success",
                "prediction": letter,
                "confidence": confidence_percent,
                "confidence_val": confidence,
                "hand_detected": True,
                "landmarks": landmarks_list
            })
        
        # Fallback when no hand detected under any strategy
        return jsonify({
            "status": "no_hand",
            "prediction": "NOTHING",
            "confidence": "0.0%",
            "confidence_val": 0.0,
            "hand_detected": False
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
