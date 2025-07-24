# ❤️ Heart Disease Prediction Platform

A comprehensive web application for predicting heart disease risk using advanced machine learning models, interactive assessments, and explainable AI. Built for healthcare professionals, researchers, and individuals seeking insights into cardiovascular health.

---

## 🚀 Features
- **Multi-Model Prediction:** KNN, Random Forest, XGBoost, and more
- **Quick Assessment:** Fast risk evaluation with minimal input
- **Explainable AI:** Understand model decisions
- **PDF & Image Uploads:** Analyze ECGs and medical reports
- **Admin Dashboard:** Manage users and reports
- **Interactive Chatbot:** Get health tips and explanations

---

## 🛠️ Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/MAsad91/CardioVascular-Disease-Predictor.git
   cd CardioVascular-Disease-Predictor
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python app.py
   ```

---

## 📁 Excluded Files & How to Generate/Attach Them
Certain files and folders are excluded from version control for security, privacy, and size reasons:

- `instance/` (Database): Contains `heart_disease.db` with user and report data. **Not included for privacy.**
  - **To generate:** The database will be created automatically on first run. For demo data, contact the maintainer.
- `models/` (Trained Models): Contains `.pkl` model files and images. **Not included due to size.**
  - **To generate:** Run the training scripts in `src/` (e.g., `train_model.py`, `train_quick_assessment_model.py`).
- `temp/` (Temporary Files): Stores temporary prediction results. **Not needed for setup.**
- `venv/` (Virtual Environment): Not included. Create your own as shown above.

> **Note:** If you have pre-trained models or a database to use, place them in the respective folders after cloning.

---

## 📊 Project Structure
```
Heart-Disease/
├── app.py
├── config.py
├── data/
├── instance/         # (excluded) Database
├── models/           # (excluded) Trained models
├── src/
├── static/
├── templates/
├── temp/             # (excluded) Temporary files
├── venv/             # (excluded) Virtual environment
└── .gitignore
```

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact
For questions, suggestions, or demo data requests, please open an issue or contact the maintainer at [your-email@example.com].

---

## 📝 License
This project is licensed under the MIT License. 