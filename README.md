# 🏥 Explainable AI (XAI) Patient Safety & Clinical Decision Support System

An end-to-end web application built for real-time patient risk monitoring, explainable machine learning predictions (SHAP), and an interactive nurse-assistant clinical chatbot. Developed as a final-year project.

---

## 🚀 Features
* **Real-Time Telemetry Tracking:** Monitor patient vitals including Heart Rate, Blood Oxygen ($\text{SpO}_2$), Systolic Blood Pressure, and Temperature.
* **Predictive Risk Assessment:** Utilizes a scikit-learn Random Forest classifier to instantly flag critical health risks.
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) values to break down and visualize *why* the model flagged a patient.
* **Nurse-Assistant Chatbot:** An interactive natural language layer that translates complex clinical metrics and risk scores into actionable insights for medical staff.

---

## 🛠️ Tech Stack
* **Frontend & UI:** Streamlit
* **Backend & Inference:** FastAPI, Python
* **Machine Learning & XAI:** Scikit-Learn, SHAP, NumPy, Pandas

---

## 📂 Project Structure
```text
xai-patient-safety/
│
├── main.py            # FastAPI backend (Model, SHAP engine, and Chatbot logic)
├── app.py             # Streamlit interactive user interface
└── requirements.txt   # Project dependencies
