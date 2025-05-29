import tensorflow as tf
import numpy as np
import yaml
import cv2

class Classifier:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = tf.keras.models.load_model(self.config['models']['classifier']['path'])
        self.confidence_threshold = self.config['models']['classifier']['confidence_threshold']
        self.preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

    def preprocess_image(self, image):
        image = cv2.resize(image, (32, 32))
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        #image = self.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image

    def classify(self, image):
        try:
            preprocessed_image = self.preprocess_image(image)
            prediction = self.model.predict(preprocessed_image)[0][0]

            if prediction >= self.confidence_threshold:
                return {'class': 'animal', 'confidence': float(prediction)}
            else:
                return None
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            return None