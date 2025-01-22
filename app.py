import streamlit as st
import time


st.set_page_config(
    page_title="ML Super App",  # This changes the display name in the browser tab
    page_icon="🚀",       # Optional: Add a custom favicon (emoji or path to an icon file)
    layout="wide"         # Optional: Use a wide layout
)



# Custom CSS for styling
st.markdown("""
    <style>
        .reportview-container {
            padding: 0;
        }
        .main {
            max-width: 100%;
        }
        .block-container {
            padding: 1rem 2rem;
        }
        .app-box {
            background-color: #f4f4f4;
            border-radius: 10px;
            padding: 15px;  /* Reduced padding */
            margin: 10px 0;  /* Reduced margin */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            height: 100%;  /* Ensure all boxes have equal height */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .stButton>button {
            font-size: 16px;
            padding: 12px 20px;
            margin-top: 10px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stColumn {
            padding: 0;
        }
        /* Reduce margins around content */
        .stText {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Function to simulate home page loading
def home_page():

        
    
    # Title
    st.title("✨ Welcome to the ML Super App! 🚀")
    
    # Project Overview
    st.markdown("""
    **ML Super App** is designed to simplify and automate key steps in the machine learning pipeline. 
    Whether you're a beginner or experienced, this app offers a variety of tools that allow you to 
    apply machine learning techniques with minimal effort.
    """)

    st.markdown("---")
    st.markdown("🚀 **Get started now by navigating to the first step from the sidebar!**")
    st.balloons()

# Display homepage content
home_page()

# Grid layout for explaining each application (without buttons)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="app-box">
            <h3>🔮 Supervised Learning</h3>
            <p>Start your supervised learning journey with popular regression and classification algorithms. This tool helps you build predictive models using labeled data.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="app-box">
            <h3>🧠 Unsupervised Learning</h3>
            <p>Explore clustering, dimensionality reduction, and unsupervised learning techniques to uncover hidden patterns and structures in your data.</p>
        </div>
    """, unsafe_allow_html=True)

# Additional tools for imbalanced datasets and feature selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="app-box">
            <h3>⚖️ Handle Imbalanced Dataset</h3>
            <p>Address issues of class imbalance in your dataset by using techniques like SMOTE, under-sampling, or over-sampling to ensure more accurate models.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="app-box">
            <h3>🔍 Input Variable Selection</h3>
            <p>Improve model performance by selecting the most important features, removing irrelevant or redundant ones to boost predictive power.</p>
        </div>
    """, unsafe_allow_html=True)

# Final instruction
st.markdown("""
    For more details and to start using these tools, please navigate through the **Sidebar**.
""")

# Credits section (now placed at the end)
st.subheader("👥 Credits:")
st.markdown("""
- **Developer**: Shaik Hidaythulla [GitHub](https://github.com/SHAIK-07) 🔗, [LinkedIn](https://www.linkedin.com/in/shaik-hidaythulla/) 🔗 
- **Technologies Used**: 
    - [Streamlit](https://streamlit.io/) for UI  
    - [Pandas Profiling](https://pandas-profiling.ydata.ai/docs/master/index.html) for data insights  
    - [Scikit-learn](https://scikit-learn.org/) for machine learning  
    - [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization  
- **Special Thanks**: To the open-source community for tools and libraries that made this possible.
""")