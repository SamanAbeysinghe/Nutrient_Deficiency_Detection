import tensorflow as tf
from PIL import Image
import numpy as np
import io


class ImageClassifier:

    def __init__(self, model_path, class_labels):
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.class_labels = class_labels

    def preprocess_image(self, image):
        image = image.resize((180, 180))
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        return np.expand_dims(image_array, axis=0)

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_array = self.preprocess_image(image)
            self.model.set_tensor(self.input_details[0]['index'], image_array)
            self.model.invoke()
            output_data = self.model.get_tensor(
                self.output_details[0]['index'])
            predicted_class_index = np.argmax(output_data)
            predicted_class_name = self.class_labels[predicted_class_index]
            return predicted_class_name
        except Exception as e:
            return str(e)
