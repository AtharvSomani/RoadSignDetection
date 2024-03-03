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

# Define a function to update the description based on the selected image
def update_description():
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

# Create a frame for better layout
frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

# Create a label for the project description
description_text = tk.StringVar()
description_label = tk.Label(frame, textvariable=description_text, wraplength=400)
description_label.pack(pady=10)

# Create an "About" section label
about_label = tk.Label(frame, text="About:", font=("Helvetica", 16, "bold"))
about_label.pack(pady=5)

# Create a label for the about section
about_text = """This application uses a Convolutional Neural Network (CNN) to detect road signs. 
It predicts the description of the detected road sign and provides translations in multiple languages."""
about_text_label = tk.Label(frame, text=about_text, wraplength=400, justify="left")
about_text_label.pack(pady=10)

# Create a button to select an image and update the description
select_button = tk.Button(frame, text="Select Image", command=update_description)
select_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
