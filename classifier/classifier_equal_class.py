import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Dataset path
dataset_path = 'anomaly_detection_dataset'
image_size = (150, 150)
batch_size = 32
val_split = 0.2

# Load and split the dataset manually
dataset = image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True
)

# Extract image and label arrays
image_data = []
labels = []

for images, lbls in dataset:
    image_data.append(images.numpy())
    labels.append(lbls.numpy())

image_data = np.concatenate(image_data)
labels = np.concatenate(labels)

# Split the dataset into train, validation, and test with equal class representation
X_train, X_temp, y_train, y_temp = train_test_split(
    image_data, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Data augmentation for training
datagen_train = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, rotation_range=20)
train_generator = datagen_train.flow(X_train, y_train, batch_size=batch_size)

datagen_val_test = ImageDataGenerator(rescale=1.0 / 255)
val_generator = datagen_val_test.flow(X_val, y_val, batch_size=batch_size)
test_generator = datagen_val_test.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

# Generate predictions for the test set
Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int)
y_true = y_test

# Plot confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Good', 'Bad']))

# Save the model
model.save('good_bad_classifier_equal_class.h5')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
