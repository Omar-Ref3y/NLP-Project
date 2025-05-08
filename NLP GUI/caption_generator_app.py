import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text, END, DISABLED, NORMAL, scrolledtext
from PIL import Image, ImageTk
import threading
import traceback

class CaptionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Caption Generator")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Set paths
        self.model_path = os.path.join("models", "CNN LSTM greedy (52%).keras")
        self.tokenizer_path = os.path.join("data", "tokenizer.pkl")
        self.features_path = os.path.join("data", "combined_features.pkl")
        
        # Initialize variables
        self.model = None
        self.tokenizer = None
        self.max_length = 51  # Set to match the model's expected input shape
        self.vgg_model = None
        self.image_path = None
        self.photo = None
        self.model_loaded = False
        
        # Create UI components
        self.create_widgets()
        
        # Load model and tokenizer in a separate thread
        self.update_status("Loading model and tokenizer...")
        threading.Thread(target=self.load_model_and_tokenizer).start()
    
    def create_widgets(self):
        # Main frame
        main_frame = Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = Label(main_frame, text="Image Caption Generator", font=("Arial", 18, "bold"), bg="#f0f0f0")
        title_label.pack(pady=(0, 20))
        
        # Image display area
        self.image_frame = Frame(main_frame, bg="white", width=400, height=300, bd=2, relief="groove")
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = Label(self.image_frame, bg="white")
        self.image_label.pack(fill="both", expand=True)
        
        # Buttons frame
        button_frame = Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        self.upload_button = Button(button_frame, text="Upload Image", command=self.upload_image, 
                                    bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.upload_button.pack(side="left", padx=5)
        
        self.generate_button = Button(button_frame, text="Generate Caption", command=self.generate_caption,
                                     bg="#2196F3", fg="white", font=("Arial", 12), padx=10, pady=5, state=DISABLED)
        self.generate_button.pack(side="left", padx=5)
        
        # Caption display
        caption_frame = Frame(main_frame, bg="#f0f0f0")
        caption_frame.pack(fill="x", pady=10)
        
        caption_label = Label(caption_frame, text="Generated Caption:", font=("Arial", 12, "bold"), bg="#f0f0f0")
        caption_label.pack(anchor="w")
        
        self.caption_text = Text(caption_frame, height=3, width=70, font=("Arial", 12), wrap="word", bd=2, relief="groove")
        self.caption_text.pack(fill="x", pady=5)
        self.caption_text.config(state=DISABLED)
        
        # Status area
        status_frame = Frame(main_frame, bg="#f0f0f0")
        status_frame.pack(fill="both", expand=True, pady=10)
        
        status_label = Label(status_frame, text="Status Log:", font=("Arial", 12, "bold"), bg="#f0f0f0")
        status_label.pack(anchor="w")
        
        # Using scrolledtext for better log viewing
        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, width=70, font=("Arial", 10), wrap="word", bd=2)
        self.status_text.pack(fill="both", expand=True, pady=5)
    
    def load_model_and_tokenizer(self):
        try:
            # Load the CNN-LSTM model
            self.update_status(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            self.update_status(f"Model loaded successfully. Input shape: {self.model.input_shape}")
            
            # Load the tokenizer
            self.update_status(f"Loading tokenizer from: {self.tokenizer_path}")
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.update_status(f"Tokenizer loaded successfully. Vocab size: {len(self.tokenizer.word_index) + 1}")
            
            # Create word to index and index to word mappings from tokenizer
            self.update_status("Creating word mappings from tokenizer")
            self.word_to_idx = self.tokenizer.word_index
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
            self.update_status(f"Word mappings created. Vocabulary size: {len(self.word_to_idx)}")
            
            # Load VGG16 model for feature extraction
            self.update_status("Loading VGG16 model for feature extraction...")
            try:
                self.vgg_model = VGG16(weights='imagenet', include_top=True)
                # Remove the classification layer
                self.vgg_model = tf.keras.Model(inputs=self.vgg_model.inputs, outputs=self.vgg_model.layers[-2].output)
                self.update_status(f"VGG16 model loaded successfully. Output shape: {self.vgg_model.output_shape}")
            except Exception as e:
                self.update_status(f"Error loading VGG16: {str(e)}")
                self.update_status("Trying alternative method to load VGG16...")
                try:
                    # Try a different approach to load VGG16
                    from tensorflow.keras.applications import VGG16
                    self.vgg_model = VGG16(weights='imagenet', include_top=True)
                    self.vgg_model = tf.keras.Model(inputs=self.vgg_model.inputs, outputs=self.vgg_model.layers[-2].output)
                    self.update_status("VGG16 loaded with alternative method")
                except Exception as e2:
                    self.update_status(f"Failed to load VGG16: {str(e2)}")
                    return
            
            # Set flag to indicate model is loaded
            self.model_loaded = True
            self.update_status("System ready. Upload an image to generate a caption.")
        except Exception as e:
            error_details = traceback.format_exc()
            self.update_status(f"Error loading model or tokenizer: {str(e)}")
            self.update_status(f"Error details: {error_details}")
    
    def update_status(self, message):
        def _update():
            self.status_text.insert(END, message + "\n")
            self.status_text.see(END)
            # Also print to console for debugging
            print(f"[STATUS] {message}")
        self.root.after(0, _update)
    
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if self.image_path:
            # Display the selected image
            self.display_image(self.image_path)
            self.update_status(f"Image loaded: {os.path.basename(self.image_path)}")
            
            # Enable generate button if model is loaded
            if self.model is not None:
                self.generate_button.config(state=NORMAL)
    
    def display_image(self, image_path):
        # Load and resize image for display
        img = Image.open(image_path)
        img = self.resize_image(img, (380, 280))
        photo = ImageTk.PhotoImage(img)
        
        # Update image in label
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def resize_image(self, img, size):
        # Resize image while maintaining aspect ratio
        width, height = img.size
        ratio = min(size[0]/width, size[1]/height)
        new_size = (int(width*ratio), int(height*ratio))
        return img.resize(new_size, Image.LANCZOS)
    
    def extract_features(self, image_path):
        try:
            # Load and preprocess the image
            self.update_status(f"Loading image from: {image_path}")
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Check if VGG model is loaded
            if self.vgg_model is None:
                self.update_status("VGG model not loaded. Trying to load it now...")
                self.vgg_model = VGG16(weights='imagenet', include_top=True)
                self.vgg_model = tf.keras.Model(inputs=self.vgg_model.inputs, outputs=self.vgg_model.layers[-2].output)
            
            # Extract features using VGG16
            self.update_status("Predicting features with VGG16...")
            features = self.vgg_model.predict(img_array, verbose=0)
            self.update_status(f"Features extracted successfully. Shape: {features.shape}")
            return features
        except Exception as e:
            error_details = traceback.format_exc()
            self.update_status(f"Error in feature extraction: {str(e)}")
            self.update_status(f"Error details: {error_details}")
            return None
    
    def generate_caption(self):
        if not self.image_path:
            self.update_status("Please upload an image first.")
            return
        
        if not self.model_loaded:
            self.update_status("Model is not fully loaded yet. Please wait.")
            return
        
        # Disable buttons during processing
        self.generate_button.config(state=DISABLED)
        self.upload_button.config(state=DISABLED)
        
        # Clear previous caption
        self.caption_text.config(state=NORMAL)
        self.caption_text.delete(1.0, END)
        self.caption_text.config(state=DISABLED)
        
        self.update_status("Extracting image features...")
        
        # Use threading to prevent UI freezing
        threading.Thread(target=self._generate_caption_thread).start()
    
    def _generate_caption_thread(self):
        try:
            # Extract features
            self.update_status("Extracting image features...")
            features = self.extract_features(self.image_path)
            if features is None:
                self.update_status("Failed to extract features from the image.")
                return
                
            self.update_status(f"Features shape: {features.shape}")
            
            # Generate caption using greedy search
            self.update_status("Generating caption...")
            caption = self.greedy_search(features)
            if caption:
                self.update_status(f"Generated caption: {caption}")
                
                # Display the generated caption
                def update_caption():
                    self.caption_text.config(state=NORMAL)
                    self.caption_text.delete(1.0, END)
                    self.caption_text.insert(END, caption)
                    self.caption_text.config(state=DISABLED)
                
                self.root.after(0, update_caption)
                self.update_status("Caption generated successfully.")
            else:
                self.update_status("Failed to generate caption.")
        except Exception as e:
            error_details = traceback.format_exc()
            self.update_status(f"Error generating caption: {str(e)}")
            self.update_status(f"Error details: {error_details}")
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.generate_button.config(state=NORMAL))
            self.root.after(0, lambda: self.upload_button.config(state=NORMAL))
    
    def greedy_search(self, features):
        try:
            # Start with the start token
            in_text = 'startseq'
            
            self.update_status(f"Starting greedy search with: '{in_text}'...")
            
            # Iterate until max length or end token
            for i in range(self.max_length):
                # Tokenize the current text
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                
                # Pad the sequence
                sequence = pad_sequences([sequence], maxlen=self.max_length)
                
                # Predict the next word
                self.update_status(f"Predicting next word (step {i+1})...")
                yhat = self.model.predict([features, sequence], verbose=0)
                
                # Get the index of the word with highest probability
                yhat_idx = np.argmax(yhat)
                
                # Convert the index to a word
                word = self.idx_to_word.get(yhat_idx, '')
                
                # Stop if we reach the end token
                if word == 'endseq' or word == '':
                    self.update_status(f"Reached end token or empty word at step {i+1}")
                    break
                    
                # Append the word to the current text
                in_text += ' ' + word
                
            # Remove the start token and clean up the caption
            caption = in_text.replace('startseq', '').strip()
            self.update_status(f"Final caption: '{caption}'")
            return caption.capitalize()
        except Exception as e:
            error_details = traceback.format_exc()
            self.update_status(f"Error in greedy search: {str(e)}")
            self.update_status(f"Error details: {error_details}")
            return None

def main():
    root = tk.Tk()
    app = CaptionGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
