import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    .stApp {
        background-image:  linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) ,url("https://hsmc.com.au/wp-content/uploads/2024/11/Diabetes.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    h1 {
        color: white;  /* Gold */
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .glow-text {
        font-size: 50px;
        color: #ffffff;
        text-align: center;
        text-shadow: 0 0 10px #00cfff, 0 0 20px #00cfff, 0 0 30px #00cfff;
        font-weight: bold;
    }
    </style>
    <div class="glow-text">Diabetes Prediction App</div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* Sidebar custom style */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 50, 70, 0.6);  /* Dark blue-ish tone */
        color: white;
    }

    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }

    /* Optional: make sidebar title/headings colored */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00171F;  /* Light cyan */
    }

    /* Optional: control scrollbar style inside sidebar */
    ::-webkit-scrollbar-thumb {
        background: #00cfff;
        border-radius: 10px;
    }
    </style>
            
""", unsafe_allow_html=True)


with st.sidebar.expander("📁 Project Intro"):
    st.markdown("- **This is a Diabetes Risk Prediction web app using an Artificial Neural Network (ANN)." \
    "It takes medical input features and predicts the likelihood of a Diabetes.**")
 

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **IBM Certifed Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of Large Language Models (LLMs)**")
    st.markdown("- **Have expertise in EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs.**")
    st.markdown("[💼Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("- **Numpy**")
    st.markdown("- **Pandas**")
    st.markdown("- **Matplotlib**")
    st.markdown("- **Seaborn**")
    st.markdown("- **Scikit Learn**")
    st.markdown("- **TensorFlow, Keras, Pickle**")
    st.markdown("- **Streamlit**")


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model():
    return load_model("diabetes_ann_model.keras")  # change file name if needed

# Load resources once
scaler = load_scaler()
model = load_keras_model()

# App title
#st.title("Diabetes Prediction App")

st.subheader("Enter Patient Details:")

# Feature inputs
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=0.0)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=140.0, value=0.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=0.0)
Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=0.0)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
Age = st.number_input("Age", min_value=0, max_value=100, value=0)

if st.button("Predict"):
    # Prepare and scale input
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    # Output
    prob = prediction  # model output (0 to 1)

    percent = prob * 100

    if percent <= 5:
        st.success(f"Risk Level: Low ({percent:.2f}%)")
        st.balloons()

    elif percent <= 20:
        st.info(f"Risk Level: Mild ({percent:.2f}%)")
        st.snow()

    elif percent <= 40:
        st.warning(f"Risk Level: Moderate — needs lifestyle changes & monitoring ({percent:.2f}%)")
        st.snow()

    else:
        st.error(f"Risk Level: High — seek medical advice ({percent:.2f}%)")
        st.snow()
