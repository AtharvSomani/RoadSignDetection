import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Load labels from CSV
csv_path = 'D:/RoadSignDetection/labels.csv'
df = pd.read_csv(csv_path)

# Load images and labels
def load_images_and_labels(dataset_path):
    images = []
    labels = []

    for class_id, sign_name in zip(df['ClassId'], df['Name']):
        class_folder = os.path.join(dataset_path, str(class_id))
        images_paths = [os.path.join(class_folder, image_file) for image_file in os.listdir(class_folder)]
        images_paths = images_paths[:min(len(images_paths), 100)]
        for image_path in images_paths:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32))
            images.append(img)
            labels.append(class_id)

    return np.array(images), np.array(labels)

dataset_path = 'D:/RoadSignDetection/traffic_Data/DATA'
images, labels = load_images_and_labels(dataset_path)
images, labels = shuffle(images, labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=58)
y_test = to_categorical(y_test, num_classes=58)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(58, activation='softmax')  # 58 classes for traffic signs
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('Road_sign_classifier_cnn.h5')
