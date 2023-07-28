import streamlit as st
from fastai.vision.all import *
from pathlib import Path
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import io

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_model(model_path):
    model_path = Path(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learn = torch.load(model_path, map_location=device)
    learn.model = learn.model.to(device)
    learn.dls.device = device
    return learn

def predict_image(image, model):
    img = PILImage.create(image)
    pred_class, _, _ = model.predict(img)
    return pred_class

def perform_eda(data_path):
    st.header("Exploratory Data Analysis")
    data = ImageDataLoaders.from_folder(data_path, valid_pct=0.2, item_tfms=Resize(460),
                                        batch_tfms=[], num_workers=4)
    st.write("Number of classes:", data.c)
    st.write("Class labels:", data.vocab)

    # Display a few sample images from each class
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(data.vocab):
        class_path = os.path.join(data_path, class_name)
        image_files = os.listdir(class_path)[:4]  # Display 4 sample images from each class
        for j, image_file in enumerate(image_files):
            plt.subplot(len(data.vocab), 4, i * 4 + j + 1)
            image_path = os.path.join(class_path, image_file)
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.axis("off")
            if j == 0:
                plt.title(class_name)
    plt.tight_layout()
    st.pyplot()

    # Plot class distribution
    num_images_per_class = [len(os.listdir(os.path.join(data_path, class_name))) for class_name in data.vocab]
    plt.figure(figsize=(10, 6))
    plt.bar(data.vocab, num_images_per_class)
    plt.xlabel("Bird Category")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    st.pyplot()

    # Display image count per class
    st.subheader("Image Count per Class")
    for i, class_name in enumerate(data.vocab):
        st.write(f"{class_name}: {num_images_per_class[i]} images")

    # Automatic check for class imbalance
    if len(num_images_per_class) > 1:
        std_dev = round(np.std(num_images_per_class), 2)
        if std_dev <= 100:
            st.success("The dataset is balanced.")
        else:
            st.warning("The dataset may have class imbalance.")
            st.write(f"Standard deviation of image counts: {std_dev}")
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Bird Image Classifier")
    st.sidebar.title("Options")

    # Select the mode (Explore, Train, or Test)
    mode = st.sidebar.selectbox("Select mode:", ("Explore", "Train", "Test"))

    # Load the trained model if in Test or Explore mode
    if mode in ["Explore", "Test"]:
        model_path = "./bird_classifier.pkl"  # Replace with the path to your saved model
        model = load_model(model_path)

    if mode == "Explore":
        # Explore the dataset used to train the model
        data_path = "./bird_images/train"  # Replace with the path to your dataset
        perform_eda(data_path)

    elif mode == "Train":
        # Train a new image classifier
        st.header("Train a New Classifier")

        # Load and preprocess the data
        data_path = "./bird_images"  # Replace with the path to your dataset
        data = ImageDataLoaders.from_folder(data_path, valid_pct=0.2, item_tfms=Resize(460),
                                            batch_tfms=[], num_workers=4)

        # Choose a pre-trained model (e.g., resnet34)
        pretrained_model = models.resnet34

        # Create the learner object
        learn = cnn_learner(data, pretrained_model, metrics=accuracy)

        # Train the model with custom hyperparameters
        epochs = st.slider("Number of epochs", 1, 20, 5)
        learn.fine_tune(epochs)

        # Save the trained model
        save_model = st.button("Save Trained Model")
        if save_model:
            learn.export("trained_model.pkl")
            st.success("Trained model saved successfully!")

    elif mode == "Test":
        # Test a pre-saved model by uploading an image
        st.header("Test Pre-saved Model")

        # Upload image for classification
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

            # Make prediction on the uploaded image
            predicted_class = predict_image(uploaded_image, model)
            st.write(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main()
