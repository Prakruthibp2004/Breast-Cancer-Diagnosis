import os
import numpy as np
import re
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

# path of images
bengin_images_path = './dataset/benign'
malignant_images_path = './dataset/malignant'
normal_images_path = './dataset/normal'
img_dir = "./dataset"
bengin_images = os.listdir(bengin_images_path)
malignant_images = os.listdir(malignant_images_path)
normal_images = os.listdir(normal_images_path)
images = bengin_images + malignant_images + normal_images

def find_mask(path):
    initial = path[:-4]
    final = path[-4:]
    return initial + '_mask'+ final

def load_data(image_dir: str, images : list, image_shape : tuple  = (256, 256)):
    '''
    load input data img & mask seperately

    Arg:
        image_dir(str): intial path for images diaractory
        images(list): list contains all images and masks name
        image_shape(tuple): image shape after transformation

    Return:
         tuple: containing all images list and masks list
    '''
    images_list = []
    masks_list = []

    for image in images:
        if 'mask' not in image:
            try:
                typ = image.split(' ')[0]
                img = plt.imread(os.path.join(img_dir, typ, image))
                msk = find_mask(image)
                mask = plt.imread(os.path.join(img_dir, typ, msk))
            except FileNotFoundError:
                continue

            img = cv2.resize(img, image_shape)
            mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_NEAREST)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            assert img.shape[:2] == image_shape, f"Got image shape: {img.shape}"
            assert mask.shape[:2] == image_shape, f"Got mask shape: {mask.shape}"
            
            images_list.append(img)
            masks_list.append(mask)

    images_array = np.array(images_list)
    masks_array = np.array(masks_list)
    print(images_array.shape, masks_array.shape)
    return images_array, masks_array

pro_images, pro_masks = load_data(img_dir, images)
print(pro_images.shape)
print(pro_masks.shape)
X_train, X_test, y_train, y_test = train_test_split(pro_images, pro_masks, test_size=0.2, random_state=42)
print("X_train.shape, X_test.shape, y_train.shape, y_test.shape=",X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Function to create convolution block in u-net Architecture

def Convolution_block(input_tensor : tf.Tensor, num_filters : int, kernel_size : tuple = (3, 3), use_batch_norm = False):
    ''' 
    perform two convolution operation with optional batch normalization
     
    Arg:
        input_tensor(tensor): input for convolution operation
        num_filters(int): number of filters for each convolution layers
        kernel_size(tuple): size of kernel for each convolution operation
        use_batch_norm(boolean): boolean input to use batch normalization 
    
    Return:
           tensor : after building convolution block with convolution layers
    '''

    # First Convolution
    x = Conv2D(filters=num_filters, kernel_size = kernel_size, padding='same', kernel_initializer = 'he_normal')(input_tensor)

    if use_batch_norm:
        x = BatchNormalization()(x)

    # Second Convolution
    x = Conv2D(filters=num_filters, kernel_size = kernel_size, padding='same', kernel_initializer = 'he_normal')(x)

    if use_batch_norm:
        x = BatchNormalization()(x)

    return x

# Function to crop & concatenate Convolution layers

def crop_concat(upsampled, skip):

    """Crop skip connection to match upsampled tensor size before concatenation."""
    up_shape = K.int_shape(upsampled)
    skip_shape = K.int_shape(skip)

    height_diff = skip_shape[1] - up_shape[1]
    width_diff  = skip_shape[2] - up_shape[2]

    if height_diff != 0 or width_diff != 0:
        skip = Cropping2D(((height_diff // 2, height_diff - height_diff // 2),
                           (width_diff // 2, width_diff - width_diff // 2)))(skip)
    return concatenate([upsampled, skip])

# Function to build U-Net Model

def Build_Unet(input_shape: tuple, num_filters=16, dropout_rate: float = 0.1, use_batch_norm: bool = True):

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

# Initialize & compile  U-Net Model
input_shape = (256, 256, 3)
model = Build_Unet(input_shape)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Comment out or fix the visualkeras part to avoid the error
# from visualkeras import layered_view
# layered_view(model)

# Callbacks 
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights = True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor = 0.5, verbose=1, min_lr = 1e-6 )
model_checkpoint = ModelCheckpoint('Unet_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 50,
    verbose= 1,
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
)

# Plot training & validation accuracy & loss
plt.subplots(1, 2, figsize = (25, 10))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.grid()
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.grid()
plt.legend()
plt.title('Loss')
plt.show()

def predict_mask(input):
    """
    Predict a segmenation of a given image & calculate the time of infrence

    Arg:
        image(Numpy Array): input image for prediction
        model(keras.Model): model going to use for prediction

    Return: Predicted Mask & Inference time
    """

    start_time = time.time()
    prediction = model.predict(np.expand_dims(input, axis=0), verbose=0)[0, :, :, 0]
    end_time = time.time()
    inference_time = end_time - start_time
    
    return prediction, inference_time

def visualize_segmented_mask(image, pred_mask, org_mask):
    """
    visualizes the image, predicted mask, and original mask

    Arg: 
        image(Array): Orginal Image which we have to segment
        pred_mask(Array): Model predicted mask
        org_mask(Array): Original mask
    """
    plt.subplots(1, 3, figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask)
    plt.title('Predicted Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(org_mask)
    plt.title('Original Mask')
    plt.show()

# Function to inference & visualise some samples of test data
def get_predictions(num_samples):
    """
    Inference & visualise some test data samples

    Arg: 
        num_samples(int): number of samples to inference & visualize
        
    """

    for _ in range(num_samples):
        index = np.random.randint(0, len(X_test))
        img = X_test[index]
        org_mask = y_test[index]
        pred_mask, inference_time = predict_mask(img)
        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        visualize_segmented_mask(img, pred_mask, org_mask)
        print(f'Time taken to inference:{inference_time}')

# visualize some predictions 
get_predictions(3)

# Generate predictions for the entire test set for evaluation
print("Generating predictions for evaluation...")
y_pred = []
for i in range(len(X_test)):
    pred_mask, _ = predict_mask(X_test[i])
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    y_pred.append(binary_mask.flatten())

y_pred = np.array(y_pred)
y_true = y_test.reshape(len(y_test), -1) > 0.5  # Flatten and binarize ground truth

# Calculate and display confusion matrix
print("Calculating confusion matrix...")
cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Background', 'Foreground'], 
            yticklabels=['Background', 'Foreground'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate and display classification report
print("Classification Report:")
print(classification_report(y_true.flatten(), y_pred.flatten(), 
                          target_names=['Background', 'Foreground']))

# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
precision = precision_score(y_true.flatten(), y_pred.flatten())
recall = recall_score(y_true.flatten(), y_pred.flatten())
f1 = f1_score(y_true.flatten(), y_pred.flatten())

print(f"\nAdditional Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Calculate IoU (Intersection over Union)
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

iou = calculate_iou(y_true.flatten(), y_pred.flatten())
print(f"IoU (Jaccard Index): {iou:.4f}")