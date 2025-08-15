![Diabetes Prediction App](https://mmi.edu.pk/wp-content/uploads/2025/07/5jpj.jpg)

# 🩺 Diabetes Prediction Web App

A machine learning-based web application that predicts whether a person has diabetes or not based on their health parameters.  
Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📌 Features
- Upload or input patient data for prediction.
- Pre-trained **Logistic Regression** (or other ML model) for classification.
- Interactive **Streamlit** UI.
- Real-time prediction output.
- User-friendly form for manual input.

---

## 📊 Dataset
We used the **Pima Indians Diabetes Dataset** from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

**Columns in dataset:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = No Diabetes, 1 = Diabetes)

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction

## Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

## Install dependencies

pip install -r requirements.txt

## ▶️ Run the APP

streamlit run app.py

## 📂 Project Structure

diabetes-prediction/
│
├── app.py                # Main Streamlit app
├── diabetes_model.pkl    # Trained ML model
├── scaler.pkl            # Feature scaler
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── data/
    └── diabetes.csv      # Dataset


## 🛠 Technologies Used

- Python
- Pandas, NumPy
- scikit-learn
- Streamlit

## 📈 Model Training

- Data preprocessing: Handling missing values, scaling features.
- Splitting dataset into training and testing sets.
- Model trained using Logistic Regression / Random Forest / SVM.
- Saved the model using pickle.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

