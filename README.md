# YogaPoseEstimation
 Kaggle Competition Project:
 https://www.kaggle.com/competitions/ukraine-ml-bootcamp-2023/overview
# Overview
This GitHub repository contains a Jupyter Notebook (`YogaPose.ipynb`) that provides a solution to a yoga pose classification competition on GitHub. The competition involves classifying different yoga asanas (poses) into six distinct classes based on images of individuals performing the poses. This project utilizes deep learning techniques to create an algorithm capable of precise classification, aiming for a higher Mean F1 score.

# Competition Description
Yoga has gained significant popularity in recent years, especially during the COVID-19 pandemic, as more people turned to yoga to maintain physical and mental well-being. Correct posture and alignment are crucial in yoga to achieve its intended benefits. This competition focuses on the classification of yoga poses by formulating pose estimation as a classification task. The goal is to classify different yoga asanas into six classes based on the posture of the person performing the asana. The evaluation metric is the Mean F1 score.

# Dataset
The dataset for this competition includes a `train.csv` file containing image IDs and corresponding class labels, as well as an `images` folder containing the training images. There are six classes based on the yoga pose. Participants are required to predict the correct class labels for the test images and submit their predictions in a CSV file (`submission.csv`).

# Data Preprocessing
The provided notebook pre-processes the data using the following steps:

Reading the training data from `train.csv`.
Splitting the data into training and validation sets.
Resizing images to a target size of 400x400 pixels.
Applying data augmentation to the training set.
Creating data generators for training and validation.
# Model Architecture
The notebook uses the Xception deep learning model with pre-trained weights. The custom classification head is added on top of the base model. The final model is compiled with the Adam optimizer and categorical cross-entropy loss function. The following layers are added:

- Global Average Pooling 2D
- Dropout layer (0.5 dropout rate)
- Dense layer with 256 units and ReLU activation
- Output layer with softmax activation for multi-class classification.
# Training
The model is trained over 25 epochs with early stopping and learning rate scheduling. The training progress is monitored using accuracy and loss metrics. Checkpoints are saved to track the best model based on validation accuracy.

# Evaluation
After training, the notebook loads the saved model and performs inference on the test images. Predictions are generated for each test image, and the final class labels are extracted. These predictions are then saved to a `submission.csv` file for submission to the competition.
The trained model achieves a private score of 86.534%.

# Results
The training and validation accuracy and loss curves are plotted to visualize the model's performance. The notebook also includes code to load the saved model and make predictions on new data.

# Usage
To use this repository, follow these steps:

Clone the repository to your local machine.
Install the required libraries specified in the notebook, typically using `pip` or `conda`.
Open and run the Jupyter Notebook (`YogaPose.ipynb`) to train the model and make predictions.
# Dependencies
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
# Author
Bohdan-Yarema Dekhtiar

# Acknowledgments
Heartful thanks to Kaggle for having hosted the competition, and to Google, namely its' ML Bootcamp Competition 2023 for Ukrainians', for actually running the competition. Particular gratitude for Armed Forces of Ukraine, without whom no work of mine would be possible.
# Contact
For any inquiries or issues regarding this repository, please contact yarema.dekhtiar@gmail.com.
