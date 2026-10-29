# model_service.py

import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

class PneumoniaDetector:

    def __init__(self, model_path: str):
        input_shape = (224, 224, 3)
        base_model = ResNet50(
            include_top=False, 
            weights=None,  
            input_shape=input_shape, 
            pooling="avg"
        )
        
        self.model = models.Sequential([
            base_model,
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])
        
        self.model.load_weights(model_path)

    def preprocess_image(self, image_bytes: bytes, img_size=(224, 224), to_grayscale=True):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: 
            raise ValueError("Invalid image file or format.")
       
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = preprocess_input(img.astype("float32"))

        return np.expand_dims(img, axis=0)

    def predict(self, image: bytes):
        processed_image = self.preprocess_image(image)
        probability = self.model.predict(processed_image, verbose=0)[0, 0]
        label = "Pneumonia" if probability > 0.53 else "Normal"

        return {"class": label, "confidence": round(float(probability), 3)}