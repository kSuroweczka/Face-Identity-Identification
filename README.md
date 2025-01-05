# **Face-Identity-Identification**

This project focuses on identifying individuals wearing sunglasses using advanced machine learning techniques. It utilizes Python and TensorFlow to process video data, detect faces, extract feature vectors (embeddings), and train a deep learning model for classification.

## **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Workflow](#workflow)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## **Overview**

This project identifies individuals wearing sunglasses by:

1. Extracting frames from video footage.
2. Detecting faces within the frames.
3. Generating feature embeddings for detected faces.
4. Training a deep learning model to classify individuals based on these embeddings.

The primary objective is to provide an accurate identification system using efficient and scalable deep learning methods.

---

## **Features**

- **Frame Extraction**: Extracts individual frames from video footage for processing.
- **Face Detection**: Locates and crops facial regions from images.
- **Embedding Extraction**: Generates feature vectors for facial recognition.
- **Model Training**: Builds and trains a classification model to identify individuals.
- **Model Testing**: Evaluates the trained model using confusion matrices, classification reports, and various metrics.

---

## **Technologies Used**

- Python
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- OpenCV

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/kSuroweczka/Face-Identity-Identification.git
   cd Face-Identity-Identification
   ```
2. Ensure you have the necessary video and image data available for processing.

## **Workflow & Usage**

Follow these steps to run the project:

1. **Prepare your data**:

   - Ensure that your video and image datasets are organized in the correct directory structure.

2. **Extract Frames**:

   - Open and run the `frames_extracting.ipynb` notebook. This will extract frames from your video footage.

3. **Detect Faces**:

   - After extracting frames, open the `face_detector.ipynb` notebook. This notebook will detect and crop faces from the extracted frames.

4. **Generate Embeddings**:

   - Run `embeddings_extracting.ipynb` to extract feature embeddings for the detected faces. The embeddings will be saved for training the model.

5. **Train the Model**:

   - Open the `model.ipynb` notebook to train the classification model. It uses the embeddings generated in the previous step. You can adjust hyperparameters during training for optimization.

6. **Test the Model**:
   - Use the `test_model.ipynb` notebook to evaluate the model. It will generate performance metrics such as confusion matrices and classification reports to assess the accuracy of the model.

By following these steps, you'll process your data, train your model, and evaluate its performance effectively.

## **Acknowledgments**

- **TensorFlow and Keras**: For providing powerful deep learning tools.
- **OpenCV**: For efficient image and video processing.
- **Scikit-learn**: For metrics and data manipulation tools.
