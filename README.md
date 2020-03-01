# Disaster Response Pipeline Project

In this repository it is saved data and code used to solve the Disaster Response Pipeline problem in Udacity Data Science Couse. 
Project consists of multiple steps: 
1. Prepared and cleaned data saved in database
2. Split data into train and test sets, create pipeline to find the best model parameters by Grid Search
3. Best model is used in web app. User can insert message and it is scored using pre-trained model. Each message is classified into one or multiple categories.  


Dataset by itself is imbalanced (water has a few examples). This may lead to not accurate model as it has not enough records for model training. 

### Installations:
For project was used Python, Jupyter Notebook, Flask. 
Required Python packages:
 - pandas, numpy, sqlalchemy, nltk, sklearn, flask.   

#### Files descriptions:
There are 3 directories and includes following files:
- data/
    
    ETL Pipeline Preparation.ipynb - data preprocessing notebook
    
    disaster_categories.csv - messages categories csv file
    
    disaster_messages.csv - messages csv file
    
    process_data.py - data preprocessing script
- models/
   train_classifier.py - model creation and evaluation script
   
   ML Pipeline Preparation.ipynb - model preparation notebook
- app/
   run.py - flask application to visualize web app


### Instructions:


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

