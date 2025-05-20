import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def load_dataset():
    """
    Load the heart disease dataset.
    Using UCI Heart Disease dataset.
    """
    # URL for the Heart Disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names based on the UCI documentation
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    try:
        # Load the dataset
        df = pd.read_csv(url, header=None, names=column_names, na_values='?')
        
        # Process the target variable - in the original dataset, >0 indicates the presence of heart disease
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to local dataset if available
        try:
            df = pd.read_csv("heart.csv")
            return df
        except:
            st.error("Could not load dataset from any source.")
            return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocess the heart disease dataset.
    
    Args:
        df: Pandas DataFrame containing the heart disease data
        
    Returns:
        X: Features DataFrame
        y: Target Series
    """
    # Check if dataframe is empty
    if df.empty:
        st.error("No data available for preprocessing.")
        return pd.DataFrame(), pd.Series()
    
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    
    # Get the target
    y = df_copy['target']
    
    # Drop the target column
    X = df_copy.drop('target', axis=1)
    
    # List of columns with missing values
    cols_with_missing = X.columns[X.isnull().any()].tolist()
    
    if cols_with_missing:
        # Impute missing values
        X[cols_with_missing] = imputer.fit_transform(X[cols_with_missing])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y
    
def perform_feature_engineering(X, method='standard', n_components=None, poly_degree=2, k_best=10):
    """
    Apply advanced feature engineering techniques to the dataset.
    
    Args:
        X: Features DataFrame
        method: Feature engineering method ('standard', 'pca', 'polynomial', 'select_k_best', 'combined')
        n_components: Number of components for PCA
        poly_degree: Polynomial degree for polynomial features
        k_best: Number of features to select with SelectKBest
        
    Returns:
        X_transformed: Transformed features
        feature_names: Names of features after transformation
    """
    if X.empty:
        st.error("No data available for feature engineering.")
        return X, X.columns
    
    if method == 'standard':
        # Standard scaling without additional transformations
        feature_names = X.columns
        return X, feature_names
    
    elif method == 'pca':
        # Perform PCA
        if n_components is None:
            n_components = min(5, X.shape[1])
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create feature names for PCA components
        feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        # Create a DataFrame with the PCA components
        X_transformed = pd.DataFrame(X_pca)
        X_transformed.columns = feature_names
        
        # Display explained variance 
        explained_variance = pca.explained_variance_ratio_
        
        return X_transformed, feature_names
    
    elif method == 'polynomial':
        # Generate polynomial features
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Generate feature names
        original_features = X.columns
        feature_names = poly.get_feature_names_out(original_features)
        
        # Create DataFrame with polynomial features
        X_transformed = pd.DataFrame(X_poly)
        X_transformed.columns = feature_names
        
        return X_transformed, feature_names
    
    elif method == 'select_k_best':
        # Select K best features
        selector = SelectKBest(f_classif, k=min(k_best, X.shape[1]))
        X_new = selector.fit_transform(X, None)  # Using None as y for now
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        # Get selected feature names
        feature_names = X.columns[selected_indices]
        
        # Create DataFrame with selected features
        X_transformed = pd.DataFrame(X_new)
        X_transformed.columns = feature_names
        
        return X_transformed, feature_names
    
    elif method == 'combined':
        # Combine polynomial features and PCA
        # First create polynomial features
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Then apply PCA to reduce dimensions
        if n_components is None:
            n_components = min(10, X_poly.shape[1])
            
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_poly)
        
        # Create feature names
        feature_names = [f'Combined_{i+1}' for i in range(n_components)]
        
        # Create DataFrame
        X_transformed = pd.DataFrame(X_transformed)
        X_transformed.columns = feature_names
        
        return X_transformed, feature_names
    
    else:
        st.warning(f"Unknown feature engineering method: {method}. Using standard features.")
        return X, X.columns
