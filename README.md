# Income Predictor App
A web application that predicts personal income based on census data.

## Description
This app uses machine learning models, Logistic Regression and K-Nearest Neighbors, to predict whether a person's income exceeds $50K per year based on their age, education, occupation, working hours, and other demographic features.

## Features
- Input personal information (age, education level, marital status, occupation, gender, race, etc.)  
- Choose between AI models (Logistic Regression or KNN) for prediction  
- Compare user data against dataset statistics  
- Visualize data analysis and model performance with interactive charts

## Installation and Running
1. Clone the repository:
- git clone https://github.com/poundvn07/income-classification.git
- cd income-classification

2. (Optional) Create and activate a virtual environment:
- python -m venv venv
- source venv/bin/activate   # Linux/macOS
- venv\Scripts\activate      # Windows

3. Install required packages:
pip install -r requirements.txt

4. Run the streamlit app:
streamlit run app.py

## Data File
adult.csv: The dataset used for training and analysis (should be placed in the same folder as app.py).

## Technologies Used
- Python
- Streamlit
- scikit-learn (Logistic Regression, KNN, SMOTE)
- Plotly (visualization)
- pandas, numpy

## Author
Pound


