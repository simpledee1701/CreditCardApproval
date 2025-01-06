import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, List, Any
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Credit Card Approval Predictor", layout="wide")

# Add the same CSS styles...

# Constants for file paths
MODEL_DIR = 'models'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.joblib')
RESULTS_PATH = os.path.join(MODEL_DIR, 'model_results.joblib')
PCA_PATH = os.path.join(MODEL_DIR, 'pca.joblib')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_data
# def load_and_process_data(_file_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict, StandardScaler]:
#     # Check if preprocessed data exists
#     if (os.path.exists(SCALER_PATH) and 
#         os.path.exists(ENCODERS_PATH) and 
#         os.path.exists(FEATURES_PATH)):
        
#         logger.info("Loading preprocessed data from disk...")
#         scaler = joblib.load(SCALER_PATH)
#         encoded_mappings = joblib.load(ENCODERS_PATH)
#         selected_features = joblib.load(FEATURES_PATH)
        
#         # Load and transform data
#         data = pd.read_csv(_file_path)
#         X = data[selected_features].copy()
#         y = data['Status'].copy()
        
#         # Apply existing transformations
#         numeric_features = ['Total_Income', 'Total_Children', 'Total_Family_Members', 
#                           'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt']
#         categorical_features = list(set(selected_features) - set(numeric_features))
        
#         for col in categorical_features:
#             X[col] = X[col].map(encoded_mappings[col])
        
#         X[numeric_features] = scaler.transform(X[numeric_features])
        
#         return X, y, selected_features, encoded_mappings, scaler
    
#     # If not, process the data and save the transformers
#     logger.info("Processing data for the first time...")
#     data = pd.read_csv(_file_path)
    
#     selected_features = [
#         'Applicant_Gender', 'Owned_Car', 'Owned_Realty', 'Total_Children',
#         'Total_Income', 'Income_Type', 'Education_Type', 'Family_Status',
#         'Housing_Type', 'Total_Family_Members', 'Applicant_Age',
#         'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt'
#     ]
    
#     X = data[selected_features].copy()
#     y = data['Status'].copy()
    
#     numeric_features = ['Total_Income', 'Total_Children', 'Total_Family_Members', 
#                       'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt']
#     categorical_features = list(set(selected_features) - set(numeric_features))
    
#     encoded_mappings = {}
#     for col in categorical_features:
#         le = LabelEncoder()
#         X[col] = le.fit_transform(X[col].astype(str))
#         encoded_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))
    
#     scaler = StandardScaler()
#     X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
#     # Save preprocessed data
#     joblib.dump(scaler, SCALER_PATH)
#     joblib.dump(encoded_mappings, ENCODERS_PATH)
#     joblib.dump(selected_features, FEATURES_PATH)
    
#     return X, y, selected_features, encoded_mappings, scaler

def load_and_process_data(_file_path: str, n_components: float = 0.95) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict, StandardScaler, PCA]:
    """
    Load and preprocess data with PCA dimensionality reduction.
    
    Args:
        _file_path (str): Path to the CSV file
        n_components (float): Number of components for PCA. If float between 0 and 1,
                            represents the variance to be explained
    
    Returns:
        Tuple containing:
        - X: Processed feature matrix with PCA applied
        - y: Target variable
        - selected_features: List of feature names
        - encoded_mappings: Dictionary of label encodings
        - scaler: Fitted StandardScaler
        - pca: Fitted PCA object
    """
    # Check if preprocessed data exists
    if (os.path.exists(SCALER_PATH) and 
        os.path.exists(ENCODERS_PATH) and 
        os.path.exists(FEATURES_PATH) and
        os.path.exists(PCA_PATH)):
        
        logger.info("Loading preprocessed data from disk...")
        scaler = joblib.load(SCALER_PATH)
        encoded_mappings = joblib.load(ENCODERS_PATH)
        selected_features = joblib.load(FEATURES_PATH)
        pca = joblib.load(PCA_PATH)
        
        # Load and transform data
        data = pd.read_csv(_file_path)
        X = data[selected_features].copy()
        y = data['Status'].copy()
        
        # Apply existing transformations
        numeric_features = ['Total_Income', 'Total_Children', 'Total_Family_Members', 
                          'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt']
        categorical_features = list(set(selected_features) - set(numeric_features))
        
        for col in categorical_features:
            X[col] = X[col].map(encoded_mappings[col])
        
        X[numeric_features] = scaler.transform(X[numeric_features])
        
        # Apply PCA transformation
        X_pca = pca.transform(X)
        X_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(X_pca.shape[1])])
        
        return X_pca, y, selected_features, encoded_mappings, scaler, pca
    
    # If not, process the data and save the transformers
    logger.info("Processing data for the first time...")
    data = pd.read_csv(_file_path)
    
    selected_features = [
        'Applicant_Gender', 'Owned_Car', 'Owned_Realty', 'Total_Children',
        'Total_Income', 'Income_Type', 'Education_Type', 'Family_Status',
        'Housing_Type', 'Total_Family_Members', 'Applicant_Age',
        'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt'
    ]
    
    X = data[selected_features].copy()
    y = data['Status'].copy()
    
    numeric_features = ['Total_Income', 'Total_Children', 'Total_Family_Members', 
                      'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt']
    categorical_features = list(set(selected_features) - set(numeric_features))
    
    # Encode categorical features
    encoded_mappings = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoded_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))
    
    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(X_pca.shape[1])])
    
    # Save preprocessed data and transformers
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoded_mappings, ENCODERS_PATH)
    joblib.dump(selected_features, FEATURES_PATH)
    joblib.dump(pca, PCA_PATH)
    
    logger.info(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]} features")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
    
    return X_pca, y, selected_features, encoded_mappings, scaler, pca

@st.cache_resource
def train_models(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    # Check if trained models exist
    if os.path.exists(RESULTS_PATH):
        logger.info("Loading trained models from disk...")
        return joblib.load(RESULTS_PATH)
    
    logger.info("Training models for the first time...")
    
    # Your existing train_models code...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grids = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000),
            'params': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'tol': [1e-4, 1e-5]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
    }
    
    results = {}
    for name, config in param_grids.items():
        try:
            grid_search = GridSearchCV(config['model'], config['params'], cv=5, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            results[name] = {
                'model': best_model,
                'accuracy': accuracy_score(y_test, y_pred),
                'cv_scores': cross_val_score(best_model, X_train, y_train, cv=5),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_data': {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)},
                'best_params': grid_search.best_params_,
                'feature_importance': get_feature_importance(best_model, feature_names)
            }
            
            if name == 'Decision Tree':
                plt.figure(figsize=(20,10))
                plot_tree(best_model, feature_names=feature_names, 
                         class_names=['Denied', 'Approved'], filled=True)
                plt.savefig('decision_tree.png', bbox_inches='tight', dpi=300)
                plt.close()
                
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    # Save trained models and results
    joblib.dump(results, RESULTS_PATH)
    
    return results

# Rest of your functions remain the same...
def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(feature_names, abs(model.coef_[0])))
    return None

# Your plotting functions and main() remain the same...
# [Previous plotting functions and main() code remains unchanged]

def plot_roc_curves(results):
    fig = go.Figure()
    for name, result in results.items():
        roc_data = result['roc_data']
        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            name=f"{name} (AUC={roc_data['auc']:.3f})"
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash', color='gray'),
        name='Random'
    ))
    
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )
    return fig

def plot_feature_importance(importance_dict):
    df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    return fig

st.markdown("""
    <style>
        .main {padding: 2rem;}
        .centered-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.75rem;
            transition: all 0.3s ease;
            border: none;
            margin: 1rem 0;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        .metric-card {
            background-color: black;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 5px solid #4CAF50;
        }
        .metric-card.denied {
            border-left-color: #ff4444;
        }
        .stTabs {
            background-color: black;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .section-header {
            padding: 1rem;
            background-color: #f1f3f5;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def plot_confusion_matrix(conf_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400
    )
    
    # Add text annotations
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(conf_matrix[i][j]),
                showarrow=False,
                font=dict(color='white' if conf_matrix[i][j] > conf_matrix.max()/2 else 'black')
            )
    
    return fig

def plot_precision_recall_curve(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name='Precision-Recall curve'
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=400
    )
    
    return fig

def plot_classification_report(report_dict):
    # Extract metrics for each class
    classes = []
    precision = []
    recall = []
    f1 = []
    support = []
    
    for k, v in report_dict.items():
        if k not in ('accuracy', 'macro avg', 'weighted avg'):
            classes.append(f'Class {k}')
            precision.append(v['precision'])
            recall.append(v['recall'])
            f1.append(v['f1-score'])
            support.append(v['support'])
    
    fig = go.Figure(data=[
        go.Bar(name='Precision', x=classes, y=precision),
        go.Bar(name='Recall', x=classes, y=recall),
        go.Bar(name='F1-Score', x=classes, y=f1)
    ])
    
    fig.update_layout(
        barmode='group',
        title='Classification Metrics by Class',
        xaxis_title='Class',
        yaxis_title='Score',
        height=400
    )
    
    return fig

def main():
    st.title("Credit Card Approval Prediction System")
    try:
        X, y, selected_features, encoded_mappings, scaler,pca = load_and_process_data('Application_Data.csv')
        results = train_models(X, y, selected_features)
        
        # Model Selection in center
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.markdown("### Model Selection")
        model_options = list(results.keys())
        selected_model = st.selectbox("Select prediction model:", model_options)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Application Form in center
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.markdown("### Application Form")
        with st.form("prediction_form"):
            # Create three columns for form inputs
            col1, col2, col3 = st.columns(3)
            
            inputs = {}
            cols = [col1, col2, col3]
            col_idx = 0
            
            for feature in selected_features:
                with cols[col_idx % 3]:
                    if feature in encoded_mappings:
                        options = list(encoded_mappings[feature].keys())
                        selected = st.selectbox(f"{feature.replace('_', ' ')}", options)
                        inputs[feature] = encoded_mappings[feature][selected]
                    else:
                        inputs[feature] = st.number_input(
                            f"{feature.replace('_', ' ')}",
                            value=0.0,
                            step=0.01 if feature == 'Total_Income' else 1.0
                        )
                col_idx += 1
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                input_df = pd.DataFrame([inputs])
                numeric_cols = ['Total_Income', 'Total_Children', 'Total_Family_Members', 
                              'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt']
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
                
                # Make prediction using only the selected model
                model = results[selected_model]['model']
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
                confidence = prob[1] if pred == 1 else prob[0]
                
                # Center the prediction result
                st.markdown("### Prediction Result")
                st.markdown(f"""
                <div class="metric-card {'approved' if pred == 1 else 'denied'}">
                    <h4>{selected_model}</h4>
                    <p>Decision: {'✅ Approved' if pred == 1 else '❌ Denied'}</p>
                    <p>Confidence: {confidence:.2%}</p>
                    <p>Best Parameters: {results[selected_model]['best_params']}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Analysis Tabs
        st.markdown("### Model Analysis")
        tabs = st.tabs(["Model Performance", "ROC Curves", "Feature Importance", "Confusion Matrix", "Classification Report", "Decision Tree"])
        
        with tabs[0]:
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.3%}",
                    'CV Score (mean)': f"{result['cv_scores'].mean():.3%}",
                    'AUC-ROC': f"{result['roc_data']['auc']:.3f}"
                })
            st.table(pd.DataFrame(comparison_data))
        
        with tabs[1]:
            st.plotly_chart(plot_roc_curves(results), use_container_width=True)
        
        with tabs[2]:
            for name, result in results.items():
                if result['feature_importance']:
                    st.subheader(f"{name} Feature Importance")
                    st.plotly_chart(plot_feature_importance(result['feature_importance']), use_container_width=True)
        
        with tabs[3]:
            st.subheader(f"Confusion Matrix - {selected_model}")
            conf_matrix = results[selected_model]['confusion_matrix']
            st.plotly_chart(plot_confusion_matrix(conf_matrix), use_container_width=True)
            
            # Add confusion matrix interpretation
            tn, fp, fn, tp = conf_matrix.ravel()
            st.markdown(f"""
            **Confusion Matrix Interpretation:**
            * True Negatives (Correctly Predicted Denials): {tn}
            * False Positives (Incorrectly Predicted Approvals): {fp}
            * False Negatives (Incorrectly Predicted Denials): {fn}
            * True Positives (Correctly Predicted Approvals): {tp}
            """)
        
        with tabs[4]:
            st.subheader(f"Classification Report - {selected_model}")
            report = results[selected_model]['classification_report']
            st.plotly_chart(plot_classification_report(report), use_container_width=True)
            
            # Add classification report interpretation
            st.markdown(f"""
            **Classification Metrics:**
            * Overall Accuracy: {report['accuracy']:.2%}
            * Macro Avg F1-Score: {report['macro avg']['f1-score']:.2%}
            * Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.2%}
            """)
        
        with tabs[5]:
            st.image('decision_tree.png')
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()