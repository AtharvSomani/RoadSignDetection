import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from googletrans import Translator
import pandas as pd

# Load the saved model
model = load_model('road_sign_classifier_cnn.h5')

# Load labels from CSV
labels_df = pd.read_csv('D:/RoadSignDetection/labels.csv')

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to translate descriptions
def translate_descriptions(description, dest_langs):
    translator = Translator()
    translations = {}
    for dest_lang in dest_langs:
        translated = translator.translate(description, dest=dest_lang)
        translations[dest_lang] = translated.text
    return translations

# Define a function to handle the button click event
def on_select_image():
    file_path = filedialog.askopenfilename(title='Select an image', filetypes=[('Image files', '*.jpg *.jpeg *.png')])
    if file_path:
        image = preprocess_image(file_path)
        y_pred = model.predict(image)
        predicted_label = np.argmax(y_pred)
        predicted_description = labels_df.loc[labels_df['ClassId'] == predicted_label, 'Name'].values[0]
        dest_langs = ['en', 'kn', 'mr', 'ja', 'hi']
        translated_descriptions = translate_descriptions(predicted_description, dest_langs)
        description_text.set(f"Predicted Description:\n{predicted_description}\n\nTranslations:")
        for lang, translated_description in translated_descriptions.items():
            description_text.set(description_text.get() + f"\n{lang}: {translated_description}")

# Create the main application window
root = tk.Tk()
root.title("Road Sign Detection")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=on_select_image)
select_button.pack(pady=20)

# Create a label to display the description
description_text = tk.StringVar()
description_label = tk.Label(root, textvariable=description_text, wraplength=400)
description_label.pack(padx=20, pady=10)

# Start the Tkinter event loop
root.mainloop()
