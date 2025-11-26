"""
SimSeg Web Interface
Based on ins.json web_interface_improvements
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Color palette for segmentation masks
COLORS = [
    [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
    [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
]

# Demo categories
CATEGORIES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]


def create_demo_segmentation(image):
    """Create a demo segmentation mask (replace with actual model inference)"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create demo mask with random segments
    mask = np.zeros((h, w), dtype=np.uint8)
    np.random.seed(42)
    
    # Create some random regions
    for i in range(5):
        x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
        x2, y2 = x1 + np.random.randint(w//4, w//2), y1 + np.random.randint(h//4, h//2)
        x2, y2 = min(x2, w), min(y2, h)
        mask[y1:y2, x1:x2] = np.random.randint(1, len(CATEGORIES))
    
    return mask


def mask_to_colored(mask):
    """Convert segmentation mask to colored image"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx, color in enumerate(COLORS):
        colored[mask == idx] = color
    
    return colored


def get_confidence_scores(mask):
    """Get confidence scores per category"""
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    
    scores = {}
    for idx, count in zip(unique, counts):
        if idx < len(CATEGORIES):
            scores[CATEGORIES[idx]] = round(count / total * 100, 2)
    
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Load image
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((288, 288))  # Resize to model input size
        
        # Get segmentation mask (demo - replace with actual model)
        mask = create_demo_segmentation(image)
        
        # Create colored mask
        colored_mask = mask_to_colored(mask)
        colored_mask_img = Image.fromarray(colored_mask)
        
        # Create overlay
        overlay = Image.blend(image, colored_mask_img, alpha=0.5)
        
        # Get confidence scores
        scores = get_confidence_scores(mask)
        
        return jsonify({
            'original': image_to_base64(image),
            'mask': image_to_base64(colored_mask_img),
            'overlay': image_to_base64(overlay),
            'scores': scores,
            'categories': list(scores.keys())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    """Compare original vs improved model (demo)"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB').resize((288, 288))
    
    # Demo: simulate original vs improved results
    np.random.seed(42)
    mask_original = create_demo_segmentation(image)
    
    np.random.seed(123)  # Different seed for "improved"
    mask_improved = create_demo_segmentation(image)
    
    colored_original = Image.fromarray(mask_to_colored(mask_original))
    colored_improved = Image.fromarray(mask_to_colored(mask_improved))
    
    overlay_original = Image.blend(image, colored_original, alpha=0.5)
    overlay_improved = Image.blend(image, colored_improved, alpha=0.5)
    
    return jsonify({
        'original_model': image_to_base64(overlay_original),
        'improved_model': image_to_base64(overlay_improved),
        'original_scores': get_confidence_scores(mask_original),
        'improved_scores': get_confidence_scores(mask_improved)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)

