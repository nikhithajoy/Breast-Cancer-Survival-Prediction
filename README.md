# Breast-Cancer-Survival-Prediction

## Project Overview
This project aims to predict the survival of breast cancer patients using the METABRIC dataset. The dataset includes clinical attributes, mRNA levels, and gene mutations for 1904 patients. We utilized various machine learning techniques to preprocess the data, select relevant features, and train predictive models. The final models are deployed as a web application using Streamlit.

## Dataset
The METABRIC dataset, published in Nature Communications, was collected by Professor Carlos Caldas and Professor Sam Aparicio. The dataset includes:
* Clinical attributes
* mRNA levels z-score
* Gene mutations

## Data Preprocessing
1. Dropping Unnecessary Columns
Dropped columns that were deemed unnecessary for prediction:
  * patient_id: Unique identifier for patients
  * cancer_type: All types are breast cancer, so it is not necessary
  * death_from_cancer: Knowing whether the death was due to cancer or other reasons does not add meaningful value
    
2. Encoding Categorical Variables
Categorical variables were converted to numerical values using Label Encoding. This step ensures that the machine learning models can process these attributes effectively.

3. Handling Missing Values
Employed KNN Imputation to handle missing values. Before applying KNN Imputation, we used Standard Scaling to ensure that the features are on the same scale.

## Feature Selection
We implemented two approaches for feature selection:

1. Clinical Attributes Only: This approach uses all clinical attributes separately for model prediction.
2. Relevant Attributes Only: We performed correlation analysis on all attributes against the target variable (overall_survival). We experimented with different threshold values (0.5, 0.3, 0.2, 0.1) to identify highly correlated attributes. Attributes with a correlation of 0.1 or above were selected for model prediction.

## Model Training
We trained various machine learning models using the selected features. The models used include:

* Logistic Regression
* SGD Classifier
* Support Vector Classifier
* K-Neighbors Classifier
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* XGB Classifier
Evaluated the models using accuracy scores and selected the best-performing models for deployment.

## Model Deployment
Why Streamlit?
Streamlit is an open-source app framework that combines front-end and back-end capabilities, making it ideal for machine learning and data science projects. It offers ease of use, real-time interaction, rich visualization, customization, and seamless integration, simplifying the development process.

## Web Application
The web application, developed using Streamlit, provides two main prediction options:

Prediction with Clinical Attributes Only: Allows users to input clinical attributes to predict overall survival.
Prediction with Relevant Attributes Only: Allows users to input a broader set of attributes, including those identified through correlation analysis.
An "About the Model" section provides detailed information about the models, including accuracy scores and model details, helping users understand and trust the predictions.
You can access the deployed web application [here](#usage).

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
~~~
git clone https://github.com/yourusername/breast-cancer-survival-prediction.git
~~~
2. Navigate to the project directory:
~~~
cd breast-cancer-survival-prediction
~~~
3. Create a virtual environment:
~~~
python -m venv venv
~~~
4. Activate the virtual environment:
* On windows:
  ~~~
  venv\Scripts\activate
  ~~~
* On macOS/Linux
  ~~~
  source venv/bin/activate
  ~~~
5. Install the required packages:
~~~
pip install -r requirements.txt
~~~

## Usage
To run the Streamlit web application:
~~~
streamlit run app.py
~~~

