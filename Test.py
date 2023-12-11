import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

img_path = "/Image_test/tets_v1.jpeg"
model_datasets = load_model("model_v1.h5")

def test(predicted_class,prediction_probability,other_class,other_probability):
    return print(f"Probabilitas prediksi {predicted_class}: {prediction_probability}"), print(f"Probabilitas prediksi {other_class}: {other_probability}")

def Prediksi_(img_path,model_datasets):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    plt.imshow(img.astype(np.uint8))
    plt.show()
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model_datasets.predict(images, batch_size=10)
    image = os.path.basename(img_path)  # Get the image filename

    return classes[0],image


if __name__ == '__main__':
    classes,image = Prediksi_(img_path,model_datasets)

    if classes[0] > 0.5:
        print(image + " - earmites")
    else:
        print(image + " - konjung")
    print(classes[0])
