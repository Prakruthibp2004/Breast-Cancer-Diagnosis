import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Cropping2D
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
import secrets
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///breast_cancer.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scans = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    original_image = db.Column(db.String(200), nullable=False)
    segmented_image = db.Column(db.String(200))
    gradcam_image = db.Column(db.String(200))
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20))
    recommendations = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# U-Net Model Architecture (Same as your training code)
def Convolution_block(input_tensor, num_filters, kernel_size=(3, 3), use_batch_norm=True):
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(input_tensor)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    return x

def crop_concat(upsampled, skip):
    up_shape = K.int_shape(upsampled)
    skip_shape = K.int_shape(skip)
    height_diff = skip_shape[1] - up_shape[1]
    width_diff = skip_shape[2] - up_shape[2]
    if height_diff != 0 or width_diff != 0:
        skip = Cropping2D(((height_diff // 2, height_diff - height_diff // 2),
                          (width_diff // 2, width_diff - width_diff // 2)))(skip)
    return concatenate([upsampled, skip])

def Build_Unet(input_shape=(256, 256, 3), num_filters=16, dropout_rate=0.1, use_batch_norm=True):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Convolution_block(inputs, num_filters, kernel_size=(3,3), use_batch_norm=use_batch_norm)
    p1 = MaxPool2D((2, 2))(c1)
    p1 = Dropout(dropout_rate)(p1)

    c2 = Convolution_block(p1, num_filters*2, kernel_size=(3,3), use_batch_norm=use_batch_norm)
    p2 = MaxPool2D((2, 2))(c2)
    p2 = Dropout(dropout_rate)(p2)

    c3 = Convolution_block(p2, num_filters*4, kernel_size=(3,3), use_batch_norm=use_batch_norm)
    p3 = MaxPool2D((2, 2))(c3)
    p3 = Dropout(dropout_rate)(p3)

    c4 = Convolution_block(p3, num_filters*8, kernel_size=(3,3), use_batch_norm=use_batch_norm)
    p4 = MaxPool2D((2, 2))(c4)
    p4 = Dropout(dropout_rate)(p4)

    # Bottleneck
    c5 = Convolution_block(p4, num_filters*16, kernel_size=(3,3), use_batch_norm=use_batch_norm)

    # Decoder
    u6 = Conv2DTranspose(num_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = crop_concat(u6, c4)
    u6 = Dropout(dropout_rate)(u6)
    c6 = Convolution_block(u6, num_filters*8, use_batch_norm=use_batch_norm)

    u7 = Conv2DTranspose(num_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = crop_concat(u7, c3)
    u7 = Dropout(dropout_rate)(u7)
    c7 = Convolution_block(u7, num_filters*4, use_batch_norm=use_batch_norm)

    u8 = Conv2DTranspose(num_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = crop_concat(u8, c2)
    u8 = Dropout(dropout_rate)(u8)
    c8 = Convolution_block(u8, num_filters*2, use_batch_norm=use_batch_norm)

    u9 = Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = crop_concat(u9, c1)  
    u9 = Dropout(dropout_rate)(u9)
    c9 = Convolution_block(u9, num_filters, use_batch_norm=use_batch_norm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs, outputs)
    return model

class BreastCancerModel:
    def __init__(self):
        self.unet_model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained U-Net model"""
        try:
            # Try to load saved model first
            if os.path.exists('Unet_best_model.keras'):
                self.unet_model = load_model('Unet_best_model.keras')
                print("Loaded pre-trained model successfully")
            else:
                # Build and compile new model (for demo)
                self.unet_model = Build_Unet(input_shape=(256, 256, 3))
                self.unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                print("Created new model architecture")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction - matches your training preprocessing"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 256x256 (same as training)
            img = cv2.resize(img, (256, 256))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def segment_image(self, image_path):
        """Perform image segmentation using U-Net"""
        try:
            img = self.preprocess_image(image_path)
            if img is None:
                return None, 0
            
            start_time = time.time()
            # Perform segmentation
            prediction = self.unet_model.predict(img, verbose=0)[0]
            inference_time = time.time() - start_time
            
            # Convert prediction to binary mask
            binary_mask = (prediction > 0.5).astype(np.uint8) * 255
            
            return binary_mask, inference_time
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None, 0
    
    def generate_gradcam(self, image_path, layer_name="conv2d_5"):
        """Generate GradCAM explanations"""
        try:
            img = self.preprocess_image(image_path)
            if img is None:
                return None
            
            # Get the layer
            try:
                grad_model = Model(
                    inputs=self.unet_model.inputs,
                    outputs=[self.unet_model.get_layer(layer_name).output, 
                            self.unet_model.output]
                )
            except:
                # Try default layer name
                layer_name = "conv2d_8" if len(self.unet_model.layers) > 8 else self.unet_model.layers[-3].name
                grad_model = Model(
                    inputs=[self.unet_model.inputs],
                    outputs=[self.unet_model.get_layer(layer_name).output, 
                            self.unet_model.output]
                )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img)
                loss = tf.reduce_mean(predictions)
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
            
            # Resize heatmap to original image size
            img_orig = cv2.imread(image_path)
            heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose heatmap on original image
            superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
            return superimposed_img
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            return None
    
    def analyze_segmentation(self, binary_mask):
        """Analyze segmentation results to predict breast cancer risk"""
        try:
            if binary_mask is None:
                return "Error", 0.0, "Unknown"
            
            # Calculate the percentage of white pixels (potential masses)
            total_pixels = binary_mask.size
            white_pixels = np.sum(binary_mask > 0)
            white_percentage = (white_pixels / total_pixels) * 100
            
            # Simple rule-based classification (replace with actual classifier)
            if white_percentage < 1:
                prediction = "Normal"
                confidence = max(0.85, 1 - white_percentage/100)
                risk_level = "Low"
            elif white_percentage < 5:
                prediction = "Benign"
                confidence = 0.75 + (white_percentage / 100)
                risk_level = "Medium"
            else:
                prediction = "Malignant"
                confidence = min(0.95, 0.7 + (white_percentage / 100))
                risk_level = "High"
            
            confidence = min(0.99, max(0.6, confidence))
            
            return prediction, round(confidence * 100, 2), risk_level
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return "Error", 0.0, "Unknown"
    
    def generate_recommendations(self, prediction, risk_level, user_age=None):
        """Generate personalized recommendations based on prediction"""
        base_recommendations = {
            "Normal": {
                "treatment": "Regular annual screening recommended. No immediate treatment needed.",
                "diet": "Maintain balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
                "medication": "No specific medication required. Continue regular health checkups.",
                "lifestyle": "Regular exercise 30 minutes/day, maintain healthy weight, limit alcohol consumption.",
                "follow_up": "Next screening in 12 months"
            },
            "Benign": {
                "treatment": "Regular monitoring every 6 months. Possible biopsy if growth observed. Consult specialist.",
                "diet": "Anti-inflammatory diet: omega-3 fatty acids, turmeric, green tea, berries, leafy greens.",
                "medication": "Consult doctor for pain management if needed. Regular mammograms essential.",
                "lifestyle": "Reduce stress through meditation/yoga, regular physical activity, avoid hormone therapy.",
                "follow_up": "Next check-up in 6 months"
            },
            "Malignant": {
                "treatment": "Immediate consultation with oncologist required. Options: surgery, chemotherapy, radiation therapy.",
                "diet": "High protein diet, antioxidant-rich foods, avoid processed foods and sugar. Consider nutritional counseling.",
                "medication": "Oncologist will prescribe targeted therapy based on cancer type and stage.",
                "lifestyle": "Stress management, join support groups, regular follow-ups, quit smoking immediately.",
                "follow_up": "Urgent specialist consultation required"
            }
        }
        
        recommendations = base_recommendations.get(prediction, base_recommendations["Normal"])
        
        # Add age-specific advice
        if user_age:
            if user_age < 40:
                recommendations["age_advice"] = "Younger patients may consider genetic counseling and more frequent screenings."
            elif user_age > 60:
                recommendations["age_advice"] = "Older patients should discuss comprehensive care including bone health."
        
        # Add risk-level specific advice
        if risk_level == "High":
            recommendations["risk_advice"] = "High risk detected. Immediate medical attention recommended."
        elif risk_level == "Medium":
            recommendations["risk_advice"] = "Moderate risk. Close monitoring and lifestyle changes advised."
        
        return recommendations

# Initialize ML model
ml_model = BreastCancerModel()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(image, folder, filename):
    """Save image to specified folder"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Ensure forward slashes for consistency
    folder = folder.replace('\\', '/')
    filepath = os.path.join(folder, filename).replace('\\', '/')
    
    # Handle both 2D and 3D images
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, image_bgr)
    else:
        # Grayscale or binary image
        cv2.imwrite(filepath, image)
    
    return filepath

def get_url_path(filepath):
    """Convert file path to URL-friendly path"""
    if filepath:
        # Convert to relative path from static folder
        if filepath.startswith('static/'):
            return filepath.replace('\\', '/')
        elif 'static' in filepath:
            # Extract path after static
            parts = filepath.replace('\\', '/').split('static/')
            if len(parts) > 1:
                return f"static/{parts[1]}"
        return filepath.replace('\\', '/')
    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', '')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            age=age,
            gender=gender
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.created_at.desc()).all()
    return render_template('dashboard.html', scans=user_scans)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{current_user.id}_{secrets.token_hex(8)}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename).replace('\\', '/')
            file.save(filepath)
            
            print(f"Original file saved: {filepath}")  # Debug log
            
            # Process image
            try:
                # Perform segmentation
                segmented_mask, seg_time = ml_model.segment_image(filepath)
                
                if segmented_mask is not None:
                    # Analyze segmentation
                    prediction, confidence, risk_level = ml_model.analyze_segmentation(segmented_mask)
                    
                    # Save segmented image
                    segmented_filename = f"segmented_{unique_filename}"
                    segmented_path = save_image(segmented_mask, app.config['UPLOAD_FOLDER'], segmented_filename)
                    print(f"Segmented file saved: {segmented_path}")  # Debug log
                    
                    # Generate GradCAM
                    gradcam_img = ml_model.generate_gradcam(filepath)
                    gradcam_path = None
                    if gradcam_img is not None:
                        gradcam_filename = f"gradcam_{unique_filename}"
                        gradcam_path = save_image(gradcam_img, app.config['UPLOAD_FOLDER'], gradcam_filename)
                        print(f"GradCAM file saved: {gradcam_path}")  # Debug log
                    
                    # Generate recommendations
                    recommendations = ml_model.generate_recommendations(
                        prediction, risk_level, current_user.age
                    )
                    
                    # Convert paths to URL-friendly format before saving to database
                    original_url_path = get_url_path(filepath)
                    segmented_url_path = get_url_path(segmented_path)
                    gradcam_url_path = get_url_path(gradcam_path)
                    
                    # Save scan to database
                    scan = Scan(
                        user_id=current_user.id,
                        filename=unique_filename,
                        original_image=original_url_path,
                        segmented_image=segmented_url_path,
                        gradcam_image=gradcam_url_path,
                        prediction=prediction,
                        confidence=confidence,
                        risk_level=risk_level,
                        recommendations=str(recommendations)
                    )
                    
                    db.session.add(scan)
                    db.session.commit()
                    
                    # Use URL-friendly paths for template rendering
                    return render_template('result.html', 
                                         result={
                                             'prediction': prediction,
                                             'confidence': confidence,
                                             'risk_level': risk_level,
                                             'segmentation_time': round(seg_time, 2),
                                             'recommendations': recommendations
                                         },
                                         original_image=original_url_path,
                                         segmented_image=segmented_url_path,
                                         gradcam_image=gradcam_url_path)
                else:
                    flash('Error processing image segmentation', 'error')
                    return redirect(request.url)
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff', 'error')
    
    return render_template('upload.html')

@app.route('/scan/<int:scan_id>')
@login_required
def view_scan(scan_id):
    scan = Scan.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
    recommendations = eval(scan.recommendations) if scan.recommendations else {}
    
    # Ensure paths are URL-friendly
    original_image = get_url_path(scan.original_image)
    segmented_image = get_url_path(scan.segmented_image)
    gradcam_image = get_url_path(scan.gradcam_image)
    
    return render_template('result.html', 
                         result={
                             'prediction': scan.prediction,
                             'confidence': scan.confidence,
                             'risk_level': scan.risk_level,
                             'recommendations': recommendations
                         },
                         original_image=original_image,
                         segmented_image=segmented_image,
                         gradcam_image=gradcam_image)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.age = request.form.get('age', type=int)
        current_user.gender = request.form.get('gender', '')
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

# API endpoints
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace('\\', '/')
        file.save(filepath)
        
        segmented_mask, seg_time = ml_model.segment_image(filepath)
        if segmented_mask is not None:
            prediction, confidence, risk_level = ml_model.analyze_segmentation(segmented_mask)
            recommendations = ml_model.generate_recommendations(prediction, risk_level, current_user.age)
            
            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'risk_level': risk_level,
                'segmentation_time': seg_time,
                'recommendations': recommendations
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(request.url)

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    init_db()
    # Create uploads directory if it doesn't exist
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    print(f"Upload directory: {os.path.abspath(upload_dir)}")
    app.run(debug=True, host='0.0.0.0', port=5000)