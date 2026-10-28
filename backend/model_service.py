import cv2
import numpy as np
from tensorflow.keras.models import load_model

class PneumoniaDetector:

    def __init__(self, model_path:str):
        self.model = load_model(model_path)

    def preprocess_image(self, image_path, img_size=(224, 224), to_grayscale=True):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")
        img = cv2.resize(img, img_size)
        if to_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        return np.expand_dims(img, axis=0)

    def predict(self, image_path:str):
        processed_image = self.preprocess_image(image_path)
        probability = self.model.predict(processed_image)[0,0]
        label = "Pneumonia" if probability > 0.53 else "Normal"

        return {"class": label, "confidence": round(probability, 3)}
