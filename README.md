# Crowd Monitoring and Alerting System using Machine Learning

## 📝 Description

This project develops a machine learning system to monitor crowd density from images and classify it into different categories. It utilizes a deep learning model based on the MobileNetV2 architecture with transfer learning to analyze crowd scenes and categorize them as **Low**, **Medium**, or **High** density. The system includes an optional interactive web interface built with Gradio for easy demonstration and testing.

---

## ✨ Features

* **Crowd Density Classification:** Classifies images into three density levels:
    * **Low:** 1-75 people
    * **Medium:** 76-400 people
    * **High:** 400+ people
* **Deep Learning Model:** Uses the efficient MobileNetV2 architecture fine-tuned for crowd analysis.
* **Robust Training:** Trained on a combined dataset from multiple standard benchmarks.
* **Data Augmentation:** Improves model generalization using techniques like rotation, zoom, and flipping.
* **Interactive Demo (Optional):** Includes a Gradio interface for uploading images and viewing real-time predictions.

---

## 💾 Dataset

The model was trained on an aggregated dataset comprising images from three standard crowd analysis benchmarks:

1.  **MALL Dataset**
2.  **ShanghaiTech Dataset (Part A & Part B)**
3.  **UCF-QNRF Dataset** (A subset of 1000 images was used)

A total of **4198 images** were combined and categorized based on ground-truth people counts into the Low, Medium, and High classes.

---

## 🛠️ Methodology

* **Architecture:** Convolutional Neural Network (CNN) based on **MobileNetV2**.
* **Transfer Learning:** Utilized weights pre-trained on ImageNet and fine-tuned the top 50 layers of the base model.
* **Custom Classifier:** Added a Flatten layer, Dense layer (256 units, ReLU), Dropout (0.5), and a final Dense output layer (3 units, Softmax).
* **Input Size:** Images resized to 224x224 pixels.
* **Training:**
    * Environment: Google Colab with GPU acceleration.
    * Optimizer: Adam (learning rate 5e-5).
    * Loss Function: Categorical Crossentropy.
    * Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
    * Class weights were used to handle dataset imbalance.

---

## 🚀 Setup and Usage

This project is designed to be run in a Google Colab environment.

### Prerequisites

* Python 3
* Google Account (for Google Colab and Google Drive)
* Required libraries (installed via pip in the notebook):
    * `tensorflow`
    * `pandas`
    * `numpy`
    * `scipy`
    * `matplotlib`
    * `scikit-learn`
    * `gradio`
    * `Pillow`

### Data Setup

1.  Upload the MALL, ShanghaiTech, and UCF-QNRF datasets to your Google Drive.
2.  Ensure the paths in **Section 2** of the notebook (`mall_source`, `shanghaitech_source`, `ucf_qnrf_source`) correctly point to the dataset locations in your Drive. The expected base path is `/content/drive/MyDrive/mini_project/dataset/`.

### Running the Notebook

1.  Open `Crowd_monitoring_and_alert_system.ipynb` in Google Colab.
2.  Ensure the runtime is set to use a **GPU accelerator** (Runtime -> Change runtime type -> Hardware accelerator -> GPU).
3.  Run the cells sequentially.
    * The notebook will mount your Google Drive, copy the datasets to the Colab environment (this may take several minutes), preprocess the data, define the model, train the model, and save the best weights as `best_crowd_model_3_classes.h5`.
    * Training progress, including accuracy and loss, will be displayed. Callbacks will save the best model and may stop training early if performance plateaus.
    * Finally, it will plot the training history and optionally launch the Gradio interface.

---

## 📊 Results

* The model achieves a validation accuracy of approximately **72.9%** (based on the final epochs shown in the training log output, specifically epoch 30).
* Training and validation accuracy/loss curves are plotted and saved as `training_history.png`.
* The best trained model weights are saved to `best_crowd_model_3_classes.h5`.

---

## 🖼️ Gradio Interface (Optional Demo)

* If **Section 9** of the notebook is run, a Gradio web interface will launch.
* A public URL will be provided, allowing you to access the interface from your browser.
* Upload an image containing a crowd scene.
* The interface will display the predicted density category (Low, Medium, High) along with confidence scores.

*(Note: The `crowd_Project_Demo.ipynb` notebook demonstrates loading the saved model and launching a similar Gradio interface with an added alerting feature.)*
