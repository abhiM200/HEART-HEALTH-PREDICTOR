import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data, load_dataset
from ml_models import train_models, evaluate_models, predict
from utils import display_team_info
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more attractive
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0083B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        border-radius: 10px;
        background-color: #F9F9F9;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .prediction-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 5px;
        border-radius: 3px;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #E03C3C;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>❤️ Heart Disease Prediction Application</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict your risk of heart disease using advanced machine learning</p>", unsafe_allow_html=True)

# Add a decorative separator
st.markdown("<hr style='height:3px;border:none;color:#FF4B4B;background-color:#FF4B4B;margin-bottom:2rem;'/>", unsafe_allow_html=True)

# Display team information in sidebar
with st.sidebar:
    display_team_info()
    st.markdown("---")
    st.subheader("Navigation")
    page = st.radio("Go to", ["Home", "Data Exploration", "Model Performance", "Prediction"])

# Load dataset
@st.cache_data
def get_data():
    return load_dataset()

data = get_data()

# Home page
if page == "Home":
    # Introduction card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#FF4B4B;'>Welcome to the Heart Disease Prediction App</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This application uses advanced machine learning algorithms to predict the risk of heart disease 
    based on various health parameters. Early detection and risk assessment can help manage 
    risk factors and prevent serious complications.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Features card
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#0083B8;'>🔍 Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Explore** heart disease dataset through interactive visualizations
        - **Compare** performance of different machine learning models
        - **Predict** your risk using your own health data
        - **Understand** the importance of different factors in heart disease prediction
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#0083B8;'>📋 How to use</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. Navigate through the tabs using the sidebar
        2. Explore the dataset in **"Data Exploration"**
        3. Check model performance in **"Model Performance"**
        4. Make predictions with your data in **"Prediction"**
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # About heart disease card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#0083B8;'>❤️ About Heart Disease</h3>", unsafe_allow_html=True)
    st.markdown("""
    Heart disease describes a range of conditions that affect your heart. It includes:
    - Blood vessel diseases like coronary artery disease
    - Heart rhythm problems (arrhythmias)
    - Heart defects you're born with (congenital heart defects)
    
    According to the World Health Organization, cardiovascular diseases are the leading cause of death globally.
    Early detection and prediction can help manage the risk factors and prevent serious complications.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display a sample of the dataset
    st.subheader("Sample Data")
    st.dataframe(data.head())
    
    # Show basic statistics
    st.subheader("Data Statistics")
    st.dataframe(data.describe())

# Data Exploration page
elif page == "Data Exploration":
    st.markdown("<h2 style='color:#FF4B4B; text-align:center;'>Data Exploration & Visualization</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2em; margin-bottom:30px;'>Analyze patterns and relationships in the heart disease dataset</p>", unsafe_allow_html=True)
    
    # Add decorative separator
    st.markdown("<hr style='height:2px;border:none;color:#FF4B4B;background-color:#FF4B4B;margin-bottom:20px;'/>", unsafe_allow_html=True)
    
    # Data info
    st.subheader("Dataset Information")
    
    # Create buffer for text output
    buffer = []
    buffer.append(f"Number of records: {data.shape[0]}")
    buffer.append(f"Number of features: {data.shape[1]}")
    buffer.append(f"Missing values: {data.isnull().sum().sum()}")
    
    for item in buffer:
        st.text(item)
    
    # Feature descriptions
    st.markdown("<h3 style='color:#0083B8;'>Feature Descriptions</h3>", unsafe_allow_html=True)
    
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (in mm Hg)',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thal': 'Thalassemia (1-3)',
        'target': 'Heart disease presence (1 = yes, 0 = no)'
    }
    
    # Create two columns for feature descriptions
    col1, col2 = st.columns(2)
    
    # Split features into two lists for display in columns
    features = list(feature_descriptions.items())
    half = len(features) // 2
    
    # Display in left column
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for feature, description in features[:half]:
            st.markdown(f"""
            <div style='margin-bottom:8px;'>
                <span style='font-weight:bold; color:#FF4B4B;'>{feature}</span>: {description}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display in right column
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for feature, description in features[half:]:
            st.markdown(f"""
            <div style='margin-bottom:8px;'>
                <span style='font-weight:bold; color:#FF4B4B;'>{feature}</span>: {description}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    # Visualization options
    viz_option = st.selectbox(
        "Select visualization",
        ["Class Distribution", "Age Distribution", "Correlation Matrix", "Feature Comparison"]
    )
    
    if viz_option == "Class Distribution":
        fig = plt.figure(figsize=(10, 6))
        target_counts = data['target'].value_counts()
        sns.countplot(x='target', data=data, palette=['#ff9999', '#66b3ff'])
        plt.title('Heart Disease Distribution')
        plt.xlabel('Target (0 = No Disease, 1 = Disease)')
        plt.ylabel('Count')
        st.pyplot(fig)
        
        # Add context
        st.markdown(f"""
        - **No Heart Disease (0)**: {target_counts[0]} patients ({target_counts[0]/len(data)*100:.1f}%)
        - **Heart Disease (1)**: {target_counts[1]} patients ({target_counts[1]/len(data)*100:.1f}%)
        """)
        
    elif viz_option == "Age Distribution":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=data, x='age', hue='target', multiple='stack', palette=['#ff9999', '#66b3ff'])
            plt.title('Age Distribution by Heart Disease Status')
            plt.xlabel('Age')
            plt.ylabel('Count')
            st.pyplot(fig)
            
        with col2:
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x='target', y='age', data=data, palette=['#ff9999', '#66b3ff'])
            plt.title('Age Distribution by Heart Disease Status')
            plt.xlabel('Target (0 = No Disease, 1 = Disease)')
            plt.ylabel('Age')
            st.pyplot(fig)
        
        # Add age statistics
        st.markdown(f"""
        **Age Statistics:**
        - **Overall Age Range**: {data['age'].min()} to {data['age'].max()} years
        - **Average Age**: {data['age'].mean():.1f} years
        - **Average Age (With Heart Disease)**: {data[data['target']==1]['age'].mean():.1f} years
        - **Average Age (Without Heart Disease)**: {data[data['target']==0]['age'].mean():.1f} years
        """)
            
    elif viz_option == "Correlation Matrix":
        fig = plt.figure(figsize=(12, 10))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
        plt.title('Feature Correlation Matrix')
        st.pyplot(fig)
        
        # Highlight strong correlations
        st.markdown("### Strong Correlations with Target")
        strong_corr = corr['target'].sort_values(ascending=False)
        for idx, val in strong_corr.items():
            if idx != 'target' and abs(val) > 0.2:
                st.markdown(f"- **{idx}**: {val:.3f}")
        
    elif viz_option == "Feature Comparison":
        col1, col2 = st.columns(2)
        
        numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = [f for f in data.columns if f not in numerical_features or f in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']]
        
        with col1:
            feature1 = st.selectbox("Select feature 1", numerical_features)
            
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x='target', y=feature1, data=data, palette=['#ff9999', '#66b3ff'])
            plt.title(f'{feature1} by Heart Disease Status')
            plt.xlabel('Target (0 = No Disease, 1 = Disease)')
            plt.ylabel(feature1)
            st.pyplot(fig)
        
        with col2:
            feature2 = st.selectbox("Select feature 2", numerical_features, index=1)
            
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x=feature1, y=feature2, hue='target', data=data, palette=['#ff9999', '#66b3ff'])
            plt.title(f'{feature1} vs {feature2} by Heart Disease Status')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            st.pyplot(fig)

# Model Performance page
elif page == "Model Performance":
    st.markdown("<h2 style='color:#FF4B4B; text-align:center;'>Model Performance Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2em; margin-bottom:30px;'>Compare different machine learning algorithms for heart disease prediction</p>", unsafe_allow_html=True)
    
    # Add decorative separator
    st.markdown("<hr style='height:2px;border:none;color:#FF4B4B;background-color:#FF4B4B;margin-bottom:20px;'/>", unsafe_allow_html=True)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models and get results
    if st.button("Train and Evaluate Models"):
        with st.spinner("Training models... This may take a minute."):
            models = train_models(X_train, y_train)
            results = evaluate_models(models, X_test, y_test)
            
            # Display results
            st.subheader("Model Performance Comparison")
            # Transform the nested dictionary to a DataFrame with models as rows
            results_df = pd.DataFrame({model: metrics for model, metrics in results.items()}).T
            
            # Find the best model based on accuracy
            results_df_sorted = results_df.sort_values('Accuracy', ascending=False)
            best_model_name = results_df_sorted.index[0]
            best_model = models[best_model_name]
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
                
            # Display model performance table
            st.dataframe(results_df)
            
            # Plot accuracy comparison
            fig = plt.figure(figsize=(10, 6))
            plt.bar(results_df.index, results_df['Accuracy'], color='skyblue')
            plt.title('Model Accuracy Comparison')
            plt.xlabel('Model')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Plot ROC curves
            st.subheader("ROC Curves")
            
            fig = plt.figure(figsize=(10, 8))
            for model_name, model_data in results.items():
                plt.plot(model_data['FPR'], model_data['TPR'], label=f"{model_name} (AUC = {model_data['AUC']:.3f})")
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Different Models')
            plt.legend(loc='lower right')
            st.pyplot(fig)
            
            # Feature importance
            if 'Random Forest' in models:
                st.subheader("Feature Importance (Random Forest)")
                
                rf_model = models['Random Forest']
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance)
                plt.title('Feature Importance from Random Forest')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("### Top 5 Most Important Features:")
                for idx, row in feature_importance.head(5).iterrows():
                    st.markdown(f"- **{row['Feature']}**: {row['Importance']:.4f}")

# Prediction page
elif page == "Prediction":
    st.markdown("<h2 style='color:#FF4B4B; text-align:center;'>Heart Disease Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2em; margin-bottom:30px;'>Enter your health information to predict the risk of heart disease</p>", unsafe_allow_html=True)
    
    # Add a decorative element
    st.markdown("<div style='text-align:center; margin-bottom:20px;'>❤️ 📊 💉 🩺 ❤️</div>", unsafe_allow_html=True)
    
    # Check if model exists, otherwise train it
    if not os.path.exists('best_model.pkl'):
        st.warning("Model not trained yet. Please go to the Model Performance page and train the models first.")
    else:
        # Load the best model
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 20, 90, 50)
            sex = st.radio("Sex", ["Male", "Female"])
            sex = 1 if sex == "Male" else 0
            
            cp_options = {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal Pain",
                3: "Asymptomatic"
            }
            cp = st.selectbox("Chest Pain Type", list(cp_options.keys()), format_func=lambda x: cp_options[x])
            
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs = 1 if fbs == "Yes" else 0
        
        with col2:
            restecg_options = {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }
            restecg = st.selectbox("Resting ECG Results", list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
            
            thalach = st.slider("Maximum Heart Rate", 70, 220, 150)
            exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
            exang = 1 if exang == "Yes" else 0
            
            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
            
            slope_options = {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }
            slope = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_options.keys()), format_func=lambda x: slope_options[x])
            
            ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
            
            thal_options = {
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect"
            }
            thal = st.selectbox("Thalassemia", list(thal_options.keys()), format_func=lambda x: thal_options[x])
        
        # Create input data dictionary
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Create button for prediction
        if st.button("Predict"):
            # Convert to dataframe
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction, probability = predict(model, input_df)
            
            # Display result
            st.markdown("<h3 class='prediction-title'>Prediction Result</h3>", unsafe_allow_html=True)
            
            if prediction[0] == 1:
                risk_color = "#FF4B4B"
                result_text = "High Risk of Heart Disease"
                prob_text = f"Probability: {probability[0]:.2%}"
                icon = "⚠️"
            else:
                risk_color = "#00CC96"
                result_text = "Low Risk of Heart Disease"
                prob_text = f"Probability: {1-probability[0]:.2%}"
                icon = "✅"
            
            # Create styled result card
            st.markdown(f"""
            <div class='prediction-card' style='background-color:{risk_color}20;'>
                <div style='font-size:3rem; margin-bottom:10px;'>{icon}</div>
                <div style='font-size:1.8rem; font-weight:bold; color:{risk_color};'>{result_text}</div>
                <div style='font-size:1.2rem; margin-top:10px;'>{prob_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display gauge chart for risk visualization
            fig = px.pie(values=[probability[0], 1-probability[0]], 
                         names=['Risk', 'Safe'], 
                         hole=0.7, 
                         color_discrete_sequence=['#ff9999', '#66b3ff'])
            
            fig.update_layout(
                annotations=[dict(text=f"{probability[0]:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig)
            
            # Interpretation
            st.subheader("Interpretation")
            st.markdown("""
            This prediction is based on the health parameters you entered. The model evaluates these factors to estimate 
            the likelihood of heart disease presence. 
            
            **Note**: This is not a medical diagnosis. Please consult a healthcare professional for proper medical advice.
            """)
            
            # Risk factors based on input
            st.markdown("<h3 style='color:#0083B8; margin-top:30px;'>Your Risk Factors</h3>", unsafe_allow_html=True)
            
            risk_factors = []
            
            if age > 55:
                risk_factors.append(("Age above 55", "Older age increases risk of heart disease"))
            if sex == 1:
                risk_factors.append(("Male gender", "Men have a higher risk of heart attacks than women"))
            if cp in [0, 1]:
                risk_factors.append(("Presence of angina", "Chest pain is a common symptom of heart disease"))
            if trestbps > 140:
                risk_factors.append(("High blood pressure", "Can damage arteries and reduce blood flow to the heart"))
            if chol > 240:
                risk_factors.append(("High cholesterol", "Can lead to plaque buildup in artery walls"))
            if fbs == 1:
                risk_factors.append(("High fasting blood sugar", "Diabetes increases the risk of heart disease"))
            if thalach < 120:
                risk_factors.append(("Low maximum heart rate", "May indicate decreased heart function"))
            if exang == 1:
                risk_factors.append(("Exercise induced angina", "Chest pain during activity suggests coronary artery disease"))
            if oldpeak > 2:
                risk_factors.append(("Significant ST depression", "Abnormal ECG finding associated with heart disease"))
            if ca > 0:
                risk_factors.append((f"Presence of {ca} colored major vessels", "Indicates extent of coronary artery disease"))
            if thal == 3:
                risk_factors.append(("Reversible defect in thalassemia", "Abnormal blood flow to the heart"))
            
            if risk_factors:
                st.markdown("<div class='card' style='background-color:#f9f9f9;'>", unsafe_allow_html=True)
                for factor, description in risk_factors:
                    st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <div style='font-weight:bold; color:#FF4B4B;'>• {factor}</div>
                        <div style='color:#555; font-size:0.9em; margin-left:15px;'>{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='card' style='background-color:#f9f9f9;'>
                    <p style='color:#00CC96; font-weight:bold;'>No significant risk factors identified from your inputs.</p>
                    <p>Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Add a footer with disclaimer
            st.markdown("""
            <div class='footer'>
                <p>© TIU CSBS 2026 Batch - All Rights Reserved</p>
                <p>Created by Abhishek, Anubha, Sahil, Altamash, and Priyanshu</p>
                <p>This tool is for educational purposes only and should not replace professional medical advice.</p>
            </div>
            """, unsafe_allow_html=True)
