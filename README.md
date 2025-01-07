# Credit Card Approval System

## Overview
The Credit Card Approval System is a comprehensive application designed to predict the approval status of credit card applications using machine learning models. The project incorporates data preprocessing, feature selection, model training, and performance evaluation, while offering an interactive web-based user interface for predictions and analysis.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical data.
- **Feature Selection**: Uses Principal Component Analysis (PCA) to optimize feature space.
- **Model Training**: Implements and tunes multiple models including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
- **Performance Metrics**: Provides accuracy, ROC curves, confusion matrices, classification reports, and feature importance.
- **Interactive UI**: Built with Streamlit for user-friendly interaction.
- **Visualization**: Includes advanced visualizations with Plotly and Matplotlib.

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Data Visualization**: Plotly, Matplotlib, Seaborn,scikit-learn
- **Data Handling**: Pandas, NumPy
- **Model Persistence**: Joblib

## How It Works
1. **Data Loading**:
   - The system loads and preprocesses application data from a CSV file.
   - Missing values are handled using imputers for numerical and categorical features.

2. **Feature Engineering**:
   - Selected features are transformed using standard scaling and label encoding.
   - PCA is applied to reduce dimensionality.

3. **Model Training**:
   - Multiple machine learning models are trained using GridSearchCV to optimize hyperparameters.
   - Models are evaluated on accuracy, cross-validation scores, and AUC-ROC.

4. **Prediction**:
   - Users can input applicant details via an interactive form.
   - The system predicts approval status and provides a confidence score.

5. **Analysis Tools**:
   - ROC curves, confusion matrices, and feature importance are visualized for detailed analysis.

## Installation
### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd credit-card-approval-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run credapproval.py
   ```
4. Access the app in your browser at `http://localhost:8501`.

## Usage
- **Model Selection**: Choose from Logistic Regression, Decision Tree, Random Forest, or Gradient Boosting models.
- **Input Data**: Fill in applicant details in the provided form.
- **View Results**: Check approval status, confidence level, and best model parameters.
- **Analyze Performance**: Use tabs to explore model performance metrics and visualizations.

## Files and Directories
- **`credapproval.py`**: Main application script.
- **`models/`**: Directory for storing trained models, scalers, and encoders.
- **`Application_Data.csv`**: Sample dataset (replace with your own).
- **`decision_tree.png`**: Saved visualization of the decision tree model.

## Future Enhancements
- Support for additional models and hyperparameter tuning.
- Integration with cloud-based data storage.
- Enhanced UI/UX with more interactive visualizations.
- Deployment on platforms like AWS or Azure.

## Acknowledgments
- **Libraries**: Scikit-learn, Streamlit, Pandas, Plotly, Matplotlib
- **Inspiration**: Financial data analysis and decision-making systems