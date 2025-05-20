import streamlit as st

def display_team_info():
    """
    Display team information in the Streamlit app.
    """
    st.sidebar.title("Heart Disease Prediction App")
    
    st.sidebar.markdown("### Created By:")
    team_members = [
        "Abhishek",
        "Anubha",
        "Sahil",
        "Altamash",
        "Priyanshu"
    ]
    
    for member in team_members:
        st.sidebar.markdown(f"- {member}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("© TIU CSBS 2026 Batch")
    st.sidebar.markdown("All Rights Reserved")

def get_heart_risk_info(risk_level):
    """
    Get information about heart disease risk based on risk level.
    
    Args:
        risk_level: Low, Moderate, or High
        
    Returns:
        Dictionary with recommendations and information
    """
    risk_info = {
        "Low": {
            "description": "Your risk of heart disease appears to be low based on the provided parameters.",
            "recommendations": [
                "Maintain a healthy diet rich in fruits, vegetables, and whole grains",
                "Regular exercise (at least 150 minutes of moderate activity per week)",
                "Annual check-ups with your healthcare provider",
                "Avoid smoking and limit alcohol consumption",
                "Manage stress through relaxation techniques"
            ],
            "color": "#5cb85c"  # Green
        },
        "Moderate": {
            "description": "You have a moderate risk of heart disease based on the provided parameters.",
            "recommendations": [
                "Schedule a follow-up with your doctor to discuss your cardiovascular health",
                "Monitor your blood pressure and cholesterol regularly",
                "Increase physical activity to at least 150-300 minutes per week",
                "Follow a heart-healthy diet (Mediterranean or DASH diet)",
                "Reduce sodium and saturated fat intake",
                "Maintain a healthy weight",
                "Consider stress reduction techniques like meditation or yoga"
            ],
            "color": "#f0ad4e"  # Yellow/Orange
        },
        "High": {
            "description": "You have a high risk of heart disease based on the provided parameters.",
            "recommendations": [
                "Consult with a cardiologist as soon as possible",
                "Follow a strict heart-healthy diet as advised by your doctor",
                "Regular monitoring of blood pressure, cholesterol, and blood sugar",
                "Maintain a consistent exercise routine as recommended by your healthcare provider",
                "Take all prescribed medications consistently",
                "Avoid smoking and secondhand smoke completely",
                "Limit alcohol consumption significantly",
                "Manage stress effectively through proven techniques",
                "Consider cardiac rehabilitation programs if recommended"
            ],
            "color": "#d9534f"  # Red
        }
    }
    
    return risk_info.get(risk_level, risk_info["Moderate"])
