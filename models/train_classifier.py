import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pickle

def load_data(database_filepath):
    """Loads cleaned data prepared for modelling.
    Args:
        database_filepath (str): path 

    Returns:
        dataframes X, Y, used for modelling

   """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('data_prep', engine)  
    X = df.message
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names 

def tokenize(text):
    """ Tokenize given messages
    Args:
        text (str): message

    Returns:
        Cleaned tokens with applied lemmatization

   """
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """ prepare pipeline and tune model parameters
    Returns:
        model with best parameters found by Grid Search

   """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = [
        {"clf": [MultiOutputClassifier(KNeighborsClassifier())], 
        'clf__estimator__n_neighbors': [3, 5, 10]
        }, 
        {"clf": [MultiOutputClassifier(DecisionTreeClassifier())], 
        'clf__estimator__max_depth': [3, 5, 10]
        },     
        {"clf": [MultiOutputClassifier(RandomForestClassifier())],
        "clf__estimator__n_estimators": [10, 50, 100],
        "clf__estimator__max_depth":[3, 5]}
    ]

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, cv = 4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate best model statistics: F1, precision, accuracy
    Args:
        model: created model
        X_test (dataframe): validation sample
        Y_test (dataframe): validation sample 
        category_names (list(str)): list of catagories
    Returns:
        prints statistics by category

   """
    
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))

def save_model(model, model_filepath):
    """ save model to the directory which is passed as argument
    Args:
        model: created model
        model_filepath (str): path
    Returns:
        None

   """    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()