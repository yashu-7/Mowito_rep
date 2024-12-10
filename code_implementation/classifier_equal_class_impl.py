import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model('models\good_bad_classifier_equal_class.h5')

dataset_path = 'anomaly_detection_dataset'
image_size = (150, 150)
batch_size = 8

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = datagen_test.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int) 
y_true = test_generator.classes  

conf_mat = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Good', 'Bad'])) 