import numpy as np
from PIL import Image
import tensorflow as tf


model = tf.keras.models.load_model('pipe_damaged_model.h5')

def test_image(path, expected_class):
    image = Image.open(path).resize((150, 150))
    image = np.array(image) / 255.0
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    prediction = model.predict(np.expand_dims(image, 0))[0][0]
    print(f"Файл: {path} | Предсказание: {prediction:.2f} | Ожидается: {expected_class}")

# Примеры тестов
test_image('img_1.png', "Повреждена")
test_image('img.png', "Нет повреждений")
test_image('0.jpg', "Повреждена")
test_image('992x496.jpg', "Нет повреждений")
test_image('1.jpg', "Нет повреждений")