from googletrans import Translator
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model_path = 'D:/RoadSignDetection/road_sign_classifier_cnn.h5'
model = load_model(model_path)

class_descriptions_path = 'D:\RoadSignDetection\labels.csv'
class_descriptions_df = pd.read_csv(class_descriptions_path)

def translate_descriptions(description, dest_langs):
    translator = Translator()
    translations = {}
    for dest_lang in dest_langs:
        translated = translator.translate(description, dest=dest_lang)
        translations[dest_lang] = translated.text
    return translations

def predict_and_translate(image_path, model, class_descriptions_df, dest_langs):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 3)
    img = img / 255.0
    prediction = model.predict(img)
    sign_class = np.argmax(prediction)
    sign_description = class_descriptions_df[class_descriptions_df['ClassId'] == sign_class]['Name'].values[0]
    translated_descriptions = translate_descriptions(sign_description, dest_langs)
    return sign_class, sign_description, translated_descriptions

image_path = 'D:\RoadSignDetection/traffic_Data\DATA/1/1_1.png'
dest_langs = ['en', 'kn', 'mr', 'ja', 'hi']

predicted_class, predicted_description, translated_descriptions = predict_and_translate(image_path, model, class_descriptions_df, dest_langs)

print(f"Predicted road sign class: {predicted_class}")
print(f"Predicted description: {predicted_description}")

for lang, translated_description in translated_descriptions.items():
    print(f"Translated description ({lang}): {translated_description}")
