import streamlit as st
import time
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import tempfile
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import streamlit as st
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set Streamlit page configuration
st.set_page_config(
    page_title="Feature Selection Application",  # Page title for browser tab
    page_icon="🔍",                             # Custom favicon
    layout="wide"                               # Use wide layout for the page
)

# Inject custom CSS for layout adjustments
st.markdown("""
    <style>
        .reportview-container { padding: 0; }
        .main { max-width: 100%; }
        .block-container { padding: 1rem 2rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = 0  # Default step to 0 (Home page)
if "data3" not in st.session_state:
    st.session_state.data3 = None  # Original uploaded dataset
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None  # Selected features for download
if "target" not in st.session_state:
    st.session_state.target = None  # Target variable

# Function to navigate between steps
def navigate(step):
    st.session_state.step = step

# Define navigation steps
steps = [
    "🏠 Home",
    "📂 Upload and Preview Dataset",
    "🎯 Select Target Variable",
    "🔍 Analyze Feature Importance",
    "✅ Feature Selection"
]



for idx, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"nav_{idx}"):
        navigate(idx)

# Define the Home Page
def home_page():

   

    # Page content
    st.title("🔍 Welcome to the Feature Selection Application!")
    st.markdown("""
    The **Feature Selection Application** helps you identify and select the most impactful features 
    in your dataset, improving the performance and interpretability of your machine learning models.
    """)

    st.subheader("✨ Key Features:")
    st.markdown("""
    - 📂 **Upload Dataset**: Upload your dataset in CSV or Excel format.
    - 🎯 **Target Variable Selection**: Choose the target variable for supervised learning tasks.
    - 🔍 **Analyze Feature Importance**: Use statistical and model-based methods for feature analysis.
    - ✅ **Feature Selection**: Automatically or manually refine your feature set.
    - 📥 **Download Processed Dataset**: Export the refined dataset.
    """)

    st.subheader("⚙️ How It Works:")
    st.markdown("""
    1. Upload your dataset in CSV or Excel format.
    2. Select the target variable for prediction or analysis.
    3. Analyze feature importance with model-based or statistical methods.
    4. Refine your feature set automatically or manually.
    5. Export the refined dataset for use in ML models.
    """)

    st.markdown("---")
    st.markdown("🚀 **Ready to get started? Use the sidebar to navigate!**")
    st.balloons()

# Step 0: Home Page
if st.session_state.step == 0:
    home_page()
    
    
def display_feature_selection_rules():
    st.title("📋 Feature Selection Rules")
    st.markdown("""
    **Important Rules for Feature Selection:**
    1. Feature selection can only be performed if there are at least 2 features (excluding the target variable) in the dataset.
    2. If you have fewer than 2 features, feature selection cannot be performed.
    3. Please complete Step 3 before proceeding to feature selection.
    """)

    # Check if there are enough features for selection
    if "data3" in st.session_state and st.session_state.data3 is not None:
        num_features = len(st.session_state.data3.columns) - 1  # Exclude target variable
        if num_features < 2:
            st.error("You need at least 2 features (excluding the target variable) to perform feature selection.")
            return False  # Cannot proceed with feature selection
        else:
            return True  # Proceed with feature selection
    else:
        st.warning("No dataset is available. Please complete previous steps to upload and process your data.")
        return False

def upload_and_preview_page():
    # Display the feature selection rules
    display_feature_selection_rules()
    st.title("📂 Step 1: Upload and Preview Dataset")
    st.markdown("Upload your dataset in CSV or Excel format and get a quick overview using automated profiling.")
    

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel format):", type=["csv", "xlsx"], key="upload_file")

    if uploaded_file is not None:
        # Start a timer for estimated loading time
        start_time = time.time()

        # Show a loading spinner while processing the file
        with st.spinner("Processing the file..."):
            try:
                # Check file extension and read the file
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.data3 = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    st.session_state.data3 = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    return  # Exit the function

                # Preview the first 5 rows of the dataset
                st.write("### Dataset Preview")
                st.dataframe(st.session_state.data3.head())

                # Display file shape and columns
                st.write(f"**Dataset Shape:** {st.session_state.data3.shape[0]} rows, {st.session_state.data3.shape[1]} columns")
                st.write("**Columns:**", ", ".join(st.session_state.data3.columns))

                

                # Button to trigger detailed EDA
                if st.button("Show Detailed EDA Report"):
                    # Generate profiling report for in-app display
                    profile_in_app = ProfileReport(
                        st.session_state.data3,
                        title="YData Profiling Report",
                        explorative=True,
                    )

                    # Save a temporary file for the downloadable report
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                        profile_in_app.to_file(tmpfile.name)
                        with open(tmpfile.name, "rb") as f:
                            st.download_button(
                                label="💾 Download Full Profiling Report",
                                data=f,
                                file_name="ydata_profiling_report.html",
                                mime="text/html",
                            )

                    # Display the profiling report in the Streamlit app
                    st_profile_report(profile_in_app)

                    # Display the elapsed time for generating the report
                    elapsed_time = time.time() - start_time
                    st.success(f"Report generated in {elapsed_time:.2f} seconds! 🎉")
                    st.balloons()
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")


# Step 1: Upload and Preview
if st.session_state.step == 1:
    upload_and_preview_page()
    
# Step 2: Target Variable Selection
def target_variable_selection_page():
    st.title("🎯 Step 2: Target Variable Selection")
    st.markdown("Select the target variable for your analysis.")
    st.subheader("📌 What are Target and Features?")
    st.write("""
    **Target Variable**  
    - The target variable (also called the dependent variable or label) is the column you want to predict or analyze.  
    - For example:
        - In a regression problem, this could be **house prices** or **sales amount**.
        - In a classification problem, this could be **spam/non-spam** or **loan approval status**.

    **Input Features**  
    - Input features (also called independent variables or predictors) are the columns used to make predictions about the target variable.  
    - They provide the model with information to identify patterns and relationships.  
    - For example:
        - In predicting house prices, input features could include **square footage**, **number of bedrooms**, and **location**.
        - For loan approval, input features might be **income**, **credit score**, and **employment history**.

    💡 **Tip**: Select features that are most relevant to your target variable for better model performance.
    """)

    # Check if dataset is available
    if "data3" not in st.session_state or st.session_state.data3 is None:
        st.warning("Please upload a dataset in Step 1 to proceed.")
        return

    # Display dataset overview for context
    st.write("### Dataset Overview")
    st.write(f"**Shape:** {st.session_state.data3.shape[0]} rows, {st.session_state.data3.shape[1]} columns")
    st.write(f"**Columns:** {', '.join(st.session_state.data3.columns)}")

    # Highlight the "Select the target variable" heading
    st.markdown(
    "<h2 style='color: blue;'> 🎯 Select the target variable:</h2>", 
    unsafe_allow_html=True
        )

    # Dropdown to select the target variable
    target_variable = st.selectbox(
        label="target variable:",
        options=st.session_state.data3.columns
        
    )


    if target_variable:
        # Validate the target variable type
        target_dtype = st.session_state.data3[target_variable].dtype
        unique_values = st.session_state.data3[target_variable].nunique()

        if target_dtype in ["object", "category"]:
            st.success(f"Target variable selected: **{target_variable}** (Categorical)")
            st.session_state.target = target_variable
        elif target_dtype in ["int64", "float64"]:
            if unique_values <= 10:
                st.success(f"Target variable selected: **{target_variable}** (Numerical - Suitable for Classification)")
            else:
                st.info(f"Target variable selected: **{target_variable}** (Numerical - Suitable for Regression)")
            st.session_state.target = target_variable
        else:
            st.error(
                f"The selected target variable **{target_variable}** has an unsupported data type ({target_dtype}). "
                "Please select a categorical or numerical column."
            )
            st.session_state.target = None

        # Provide details about the selected target variable
        st.write("### Target Variable Details")
        st.write(f"- **Name**: {target_variable}")
        st.write(f"- **Data Type**: {target_dtype}")
        st.write(f"- **Unique Values**: {unique_values}")
        if unique_values <= 10:
            st.write(f"- **Unique Values (Sample)**: {st.session_state.data3[target_variable].unique().tolist()}")

# Step 2: Target Variable Selection
if st.session_state.step == 2:
    target_variable_selection_page()
    

# Step 3: Display Feature Importance


def feature_importance_page():
    st.title("📊 Step 3: Feature Importance Analysis")
    st.markdown("Analyze the importance of features using various methods.")
    st.subheader("🔗 Correlation Analysis (Numerical Features)")
    st.write("""
    Correlation analysis helps identify relationships between numerical features in the dataset. By calculating the correlation coefficient (e.g., Pearson or Spearman), you can determine the strength and direction of the relationship between pairs of variables.
    
    - **Positive Correlation**: Both variables increase or decrease together (e.g., height and weight).
    - **Negative Correlation**: One variable increases while the other decreases (e.g., temperature and heating bill).
    - **Zero or Low Correlation**: No significant relationship between variables.
    
    **Use Case**: Correlation analysis can help identify redundant features or pairs of features that provide similar information, which could be useful for feature selection and dimensionality reduction.
    """)

    st.subheader("📈 Mutual Information (Categorical Features)")
    st.write("""
    Mutual information is a method used to measure the dependence between two variables, specifically suitable for categorical data. It quantifies how much information is shared between the feature and the target variable.
    
    - **High Mutual Information**: A feature provides significant information about the target, and including it in the model could improve prediction accuracy.
    - **Low Mutual Information**: The feature is less relevant to the target and might be dropped or used with caution.

    **Use Case**: Mutual information helps identify which categorical features have the most predictive power for the target variable, making it useful for feature selection in classification tasks.
    """)

    st.subheader("🌲 Model-Based Feature Importance (Random Forest)")
    st.write("""
    Random Forest models can provide insights into feature importance by evaluating how useful each feature is for making accurate predictions. Features that contribute significantly to the decision-making process of trees in the forest are considered more important.
    
    - **Higher Importance**: Features that are frequently used for splitting nodes in decision trees, leading to better model accuracy.
    - **Lower Importance**: Features that have minimal influence on model predictions.

    **Use Case**: Random Forest-based feature importance is particularly useful in identifying important features when working with complex datasets, where some features may have non-linear relationships with the target variable. It's valuable for feature selection and reducing overfitting by removing less important features.
    """)


    # Dependency checks
    if "data3" not in st.session_state or st.session_state.data3 is None:
        st.warning("Please upload a dataset in Step 1 to proceed.")
        return

    if "target" not in st.session_state or st.session_state.target is None:
        st.warning("Please select a target variable in Step 2 to proceed.")
        return

    data = st.session_state.data3
    target = st.session_state.target

    # Split features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Check if target is categorical or numerical
    target_is_categorical = y.dtype in ["object", "category"] or y.nunique() <= 10

    # Encode the target variable if categorical
    if target_is_categorical:
        y_encoded = y.astype("category").cat.codes
    else:
        y_encoded = y

    # Correlation Analysis for numerical features
    st.subheader("🔗 Correlation Analysis (Numerical Features)")
    numerical_features = X.select_dtypes(include=["int64", "float64"])

    if not numerical_features.empty and not target_is_categorical:
        correlation_matrix = numerical_features.corrwith(y, method="pearson")
        st.write("### Correlation with Target Variable")
        st.bar_chart(correlation_matrix)
    else:
        st.write("Correlation analysis is only applicable for numerical features with a numerical target.")

    # Mutual Information for Categorical Features
    st.subheader("📈 Mutual Information (Categorical Features)")
    categorical_features = X.select_dtypes(include=["object", "category"])

    if not categorical_features.empty:
        # Encode categorical features
        cat_encoded = pd.get_dummies(categorical_features, drop_first=True)

        if target_is_categorical:
            mi_scores = mutual_info_classif(cat_encoded, y_encoded, discrete_features=True)
        else:
            mi_scores = mutual_info_regression(cat_encoded, y_encoded)

        mi_scores_df = pd.DataFrame({"Feature": cat_encoded.columns, "Mutual Information": mi_scores})
        mi_scores_df = mi_scores_df.sort_values(by="Mutual Information", ascending=False)

        st.write("### Mutual Information Scores")
        st.dataframe(mi_scores_df)

        # Plot mutual information scores
        fig, ax = plt.subplots()
        sns.barplot(data=mi_scores_df.head(10), x="Mutual Information", y="Feature", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No categorical features available for mutual information analysis.")

    # Model-Based Feature Importance using Random Forest
    st.subheader("🌲 Model-Based Feature Importance (Random Forest)")
    try:
        # Handle categorical data by encoding
        X_encoded = pd.get_dummies(X, drop_first=True)

        if target_is_categorical:
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)

        model.fit(X_encoded, y_encoded)

        # Get feature importances
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({"Feature": X_encoded.columns, "Importance": importances})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Step 3: Store Feature Importance and Names
        st.session_state.feature_importance = feature_importance_df['Importance'].values
        st.session_state.feature_names = feature_importance_df['Feature'].values

        st.write("### Random Forest Feature Importances")
        st.dataframe(feature_importance_df)

        # Plot feature importances
        fig, ax = plt.subplots()
        sns.barplot(data=feature_importance_df.head(10), x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while calculating feature importances: {e}")


# Step 3: Display Feature Importance
if st.session_state.step == 3:
    
    feature_importance_page()




import streamlit as st

def manual_feature_selection_page():
    st.title("🔧 Manual Feature Selection")

    # Check if feature importances exist in session state
    if "feature_importance" not in st.session_state or "feature_names" not in st.session_state:
        st.warning("Please complete Step 3 to calculate feature importances.")
        return

    # Retrieve feature importance and feature names from session state
    feature_importance = st.session_state.feature_importance
    feature_names = st.session_state.feature_names

    # Ask the user to set a threshold for importance
    threshold = st.slider("Set the feature importance threshold:", 0.0, 1.0, 0.01, 0.01)
    st.write(f"Selected Threshold: {threshold}")

    # Filter features based on the threshold
    filtered_features = [
        (feature, importance) for feature, importance in zip(feature_names, feature_importance)
        if importance >= threshold
    ]

    if filtered_features:
        st.write("### Features above the importance threshold:")
        # Display the filtered features with their importance scores
        

        # Let the user manually select the features
        selected_features = []
        for feature, importance in filtered_features:
            checkbox = st.checkbox(f"{feature} (Importance: {importance:.4f})", value=True)
            if checkbox:
                selected_features.append(feature)

        st.session_state.selected_features = selected_features

        st.write(f"Selected features: {selected_features}")

        # Finalize Manual Feature Selection
        if st.button("Finalize Manual Selection"):
            if selected_features:
                st.session_state.data3 = st.session_state.data3[selected_features + [st.session_state.target]]
                st.success("Manual feature selection finalized.")
            else:
                st.warning("No features selected. Please select at least one feature.")
    else:
        st.write("No features meet the selected importance threshold.")



def automatic_feature_selection_page():
    st.title("🛠️ Automatic Feature Selection")
    st.markdown("Automatically select features using techniques like RFE, Variance Threshold, etc.")

    # Check the number of features in the dataset
    num_features = len(st.session_state.data3.columns) - 1  # Exclude target variable

    # Handle case when there are less than 2 features (for RFE)
    if num_features < 2:
        st.error("You need at least 2 features (excluding the target variable) to perform feature selection.")
        return
    
    st.markdown(
    "<h2 style='color: blue;'>🔍  Select feature selection method:</h2>", 
    unsafe_allow_html=True
        )

    # Select feature selection method
    method = st.selectbox(
        label="selection methods",
        options=["RFE", "Variance Threshold"]
    )

    # Adjust the slider range dynamically based on the number of features
    n_features = st.slider(
        "Select the number of features to select",
        min_value=1,
        max_value=num_features,
        value=min(1, num_features)  # Ensure the default value is valid
    )

    # Preprocess data to handle categorical features
    data = st.session_state.data3.drop(columns=[st.session_state.target])
    
    # Encode categorical columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])

    # Perform feature selection based on the chosen method
    if method == "RFE":
        # Perform RFE using RandomForestClassifier
        rfe_selector = RFE(RandomForestClassifier(), n_features_to_select=n_features)
        rfe_selector.fit(data, st.session_state.data3[st.session_state.target])
        selected_columns = data.columns[rfe_selector.support_].to_list()

    elif method == "Variance Threshold":
        # Perform Variance Threshold
        selector = VarianceThreshold()
        selector.fit(data)

        # Get variances and sort features by variance
        variances = selector.variances_
        feature_variance_df = pd.DataFrame({"Feature": data.columns, "Variance": variances})
        feature_variance_df = feature_variance_df.sort_values(by="Variance", ascending=False)

        # Select the top n features based on variance
        selected_columns = feature_variance_df.head(n_features)["Feature"].tolist()

    st.write(f"Selected Features using {method}:")
    st.write(selected_columns)

    if st.button("Finalize Automatic Selection"):
        st.session_state.selected_features = selected_columns
        st.session_state.data3 = st.session_state.data3[selected_columns + [st.session_state.target]]
        st.success("Automatic feature selection finalized.")



def final_feature_selection_page():
    st.title("📥 Final Step : Feature Selection")
    st.markdown("Choose between Manual or Automatic Feature Selection.")

    st.subheader("🔧 Manual Feature Selection (Feature Importance Threshold)")
    st.write("""
    Manual feature selection involves the user deciding which features to keep or discard based on their understanding of the domain, dataset, and problem at hand. One common method used in manual feature selection is **Feature Importance Thresholding**, where features are kept or removed based on their importance scores.

    - **Feature Importance Threshold**:  
      - Calculate the importance of each feature using methods such as Random Forest or other model-based techniques.
      - Set a threshold for feature importance (e.g., keep features with importance > 0.05).
      - Discard features with importance below the threshold.
    
    **Use Case**: This method is useful when you want to remove less important features based on model-specific importance, improving both model performance and interpretability.
    """)

    st.subheader("🛠️ Automatic Feature Selection")
    st.write("""
    Automatic feature selection utilizes algorithms and statistical methods to automatically determine which features are most important for the model's performance. These methods can help reduce the number of features while maintaining or improving model performance.

    - **RFE (Recursive Feature Elimination)**:
      - A wrapper method that recursively removes features and builds a model on the remaining features. The model is evaluated at each iteration, and the least important features are removed until the optimal set of features is identified.
      - RFE can be used with any model and is particularly useful when working with models where feature importance is not readily available.

    - **Variance Threshold**:
      - A filter method that removes features with low variance, assuming that features with very little variance are less informative and don’t contribute much to the model's predictive power.
      - The threshold for variance can be set (e.g., remove features where variance < 0.01).
    
    **Use Case**: Automatic feature selection methods, such as RFE and Variance Threshold, are particularly useful when working with datasets with many features. RFE helps identify the most significant features, while Variance Threshold quickly removes features that add little value, improving efficiency.
    """)




    st.markdown(
    "<h2 style='color: blue;'>⚙️ Choose feature selection method:</h2>", 
    unsafe_allow_html=True
        )
    method = st.selectbox(
        label="feature selection methods:",
        options=["None", "Manual", "Automatic"]
    )

    if method == "Manual":
        manual_feature_selection_page()

    elif method == "Automatic":
        automatic_feature_selection_page()

    # Check if the dataset is available
    if "data3" in st.session_state and st.session_state.data3 is not None:
        # Display and allow download of the finalized dataset
        st.write("### Preview of Selected Dataset")
        st.dataframe(st.session_state.data3.head())
        st.download_button(
            label="Download Selected Dataset",
            data=st.session_state.data3.to_csv(index=False),
            file_name="selected_features.csv",
            mime="text/csv"
        )
        
    else:
        # Display a friendly message if data is not available
        st.warning("No dataset is available for preview. Please complete the previous steps to upload and process your data.")


# Step 6: Final Feature Selection
if st.session_state.step == 4:
    final_feature_selection_page()