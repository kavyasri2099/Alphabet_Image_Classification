# Alphabet_Image_Classification

`Table of Contents`
- Introduction
- Dataset
- Installation
- Usage
- Results
- Insights
- Introduction
  
This project focuses on classifying images of alphabets (A-Z) using various machine learning algorithms. The goal is to evaluate the performance of different models and identify the one with the highest accuracy. The algorithms used include:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
  
`Dataset`
The dataset consists of 28x28 pixel grayscale images of alphabets from A to Z. The images are organized into folders labeled with the corresponding alphabet, extracted from a ZIP file.

`Installation`
To run this project, ensure you have the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- PIL
- zipfile
  
Install the required libraries using pip:
- pip install numpy pandas matplotlib seaborn scikit-learn tqdm pillow
  
`Usage`
- Mount Google Drive and Extract Dataset: 
Ensure your dataset is accessible and properly extracted.

`Preprocess and Visualize Data`:
- Load and preprocess the images.
- Visualize the data for better understanding.
  
`Split Data and Train Models`:
- Split the dataset into training and testing sets.
- Train the models using the training set.
  
`Results`
The accuracy of each model is as follows:

- Logistic Regression: 81.90%
- Decision Tree: 71.98%
- SVM: 93.40%
- Random Forest: 90.76%
- KNN: 89.30%
  
`Insights`
Among the evaluated models, the Support Vector Machine (SVM) achieved the highest accuracy of 93.40%. Thus, the SVM is the most effective model for this classification task based on accuracy.

