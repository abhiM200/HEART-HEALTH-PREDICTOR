import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def train_models(X_train, y_train):
    """
    Train multiple machine learning models on the training data.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        models: Dictionary of trained models
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train each model
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        results: Dictionary of evaluation metrics for each model
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        
        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc_score,
            'FPR': fpr,
            'TPR': tpr
        }
    
    return results

def predict(model, input_data):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained machine learning model
        input_data: DataFrame with features for prediction
        
    Returns:
        predictions: Predicted class (0 or 1)
        probabilities: Probability of positive class
    """
    # Make prediction
    prediction = model.predict(input_data)
    
    # Get probability
    probability = model.predict_proba(input_data)[:, 1]
    
    return prediction, probability
