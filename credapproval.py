import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            margin-top: 2rem;
        }
        .stButton>button:hover {
            background-color: #FF2B2B;
        }
        .reportview-container {
            background: #f0f2f6;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        h3 {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
        }
        /* Updated styling for input labels */
        .stSelectbox label, .stNumberInput label {
            color: white !important;
            font-weight: 500;
        }
        /* Ensure input text remains visible */
        .stSelectbox > div > div[data-baseweb="select"] > div {
            color: black;
        }
        .stNumberInput > div > div > input {
            color: black;
        }
        /* Style for select dropdown options */
        div[data-baseweb="select"] > div {
            background-color: #262730;
        }
        /* Card background style */
        .card-background {
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 0.5rem;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    data = pd.read_csv('Application_Data.csv')
    
    # Clean column names and strip whitespaces
    data.columns = data.columns.str.strip()
    data = data.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    encoded_mappings = {}
    
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
        encoded_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))
    
    # Define features and target
    X = data.drop(['Applicant_ID', 'Status'], axis=1)
    y = data['Status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test, categorical_cols, encoded_mappings, scaler, pca

# Train models
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

def plot_confusion_matrix(cm, model_name, key):
    colors = px.colors.sequential.Plasma
    fig = px.imshow(cm, 
                    text_auto=True,
                    color_continuous_scale=colors,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Denied', 'Approved'],
                    y=['Denied', 'Approved'])
    
    fig.update_layout(
        title_text=f"Confusion Matrix - {model_name}",
        title_x=0.3,
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=50, r=50),
    )
    st.plotly_chart(fig, key=key, use_container_width=True)

def plot_model_comparison(results):
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in results]
    
    fig = go.Figure([
        go.Bar(
            x=model_names,
            y=accuracies,
            text=[f"{acc:.2%}" for acc in accuracies],
            textposition='auto',
            marker_color=['#FF4B4B', '#FF8C8C', '#FFB4B4'],
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Model Performance Comparison",
            x=0.3,
            font=dict(size=20)
        ),
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=50, r=50),
    )
    
    st.plotly_chart(fig, key="unique_model_comparison", use_container_width=True)

def main():
    st.title("💳 Credit Card Approval Prediction")
    
    try:
        # Load data and train models
        X, y, X_train, X_test, y_train, y_test, categorical_cols, encoded_mappings, scaler, pca = load_and_process_data()
        results = train_models(X_train, y_train, X_test, y_test)
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div class="card-background" style="background-color:white;margin-bottom: 2rem;">
                    <h3 style="margin-top: 0; color:black;">Application Details</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Collect user input
            inputs = {}
            for col in X.columns:
                if col in categorical_cols:
                    display_options = list(encoded_mappings[col].keys())
                    selected_value = st.selectbox(
                        f"{col.replace('_', ' ').title()}",
                        display_options,
                        key=f"input_{col}"
                    )
                    inputs[col] = encoded_mappings[col][selected_value]
                else:
                    inputs[col] = st.number_input(
                        f"{col.replace('_', ' ').title()}",
                        value=0.0,
                        key=f"input_{col}"
                    )
            
            if st.button("Predict Approval Status"):
                # Process input
                input_df = pd.DataFrame([inputs])
                input_scaled = scaler.transform(input_df)
                input_pca = pca.transform(input_scaled)
                
                # Get Random Forest model (best performer)
                rf_model = results['Random Forest']['model']
                
                # Predict using the Random Forest model
                prediction = rf_model.predict(input_pca)
                probability = rf_model.predict_proba(input_pca)[0]
                
                result_text = "Approved" if prediction[0] == 1 else "Denied"
                confidence = probability[1] if prediction[0] == 1 else probability[0]
                
                st.markdown(f"""
                    <div style="
                        background-color: {'#4CAF50' if result_text == 'Approved' else '#FF4B4B'};
                        color: white;
                        padding: 1rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        margin-top: 1rem;
                    ">
                        <h2 style="margin: 0;">Prediction: {result_text}</h2>
                        <p style="margin: 0.5rem 0 0 0;">Confidence: {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card-background" style="background-color:white;">
                    <h3 style="margin-top: 0;color:black;">Model Performance Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Model comparison plot
            plot_model_comparison(results)
            
            # Tabs for confusion matrices
            tabs = st.tabs(list(results.keys()))
            for tab, (name, result) in zip(tabs, results.items()):
                with tab:
                    plot_confusion_matrix(result['confusion_matrix'], name, key=f"conf_matrix_{name}")
    
    except Exception as e:
        st.error(f"""
            Error loading data or training models. Please ensure:
            1. The 'Application_Data.csv' file is in the same directory as this script
            2. The file contains the expected columns
            3. You have all required dependencies installed
            
            Error details: {str(e)}
        """)

if __name__ == '__main__':
    main()