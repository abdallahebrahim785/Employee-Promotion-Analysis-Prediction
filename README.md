# 📊 Employee Promotion Prediction System

An intelligent machine learning system that predicts employee promotion probability and provides comprehensive analytics through an interactive Streamlit dashboard.

![Dashboard Preview](Employee_Promotion.jpg)

## 🎯 Project Overview

This project analyzes employee data to identify key factors influencing promotions and builds a predictive model to forecast promotion likelihood. The system helps HR departments make data-driven decisions and understand promotion patterns.

### Key Features
- 📈 Interactive data analytics dashboard
- 🤖 Machine Learning prediction engine
- 📊 Comprehensive visualizations
- 🔍 Real-time filtering capabilities
- 💡 Actionable insights for HR decisions

## 📁 Dataset

- **Total Records:** 54,808 employees
- **Features:** 13 columns
- **Target Variable:** Promotion status (Binary classification)
- **Class Distribution:** 
  - Not Promoted: 89.8%
  - Promoted: 10.2%

### Features Used
- Department
- Region
- Education Level
- Gender
- Recruitment Channel
- Number of Trainings
- Age
- Previous Year Rating
- Length of Service
- Average Training Score

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **Joblib** - Model serialization

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/employee-promotion-prediction.git
   cd employee-promotion-prediction
```

2. **Create virtual environment**
```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the application**
```bash
   streamlit run employee_promotion_app.py
```

## 📊 Dashboard Features

### Tab 1: Executive Overview
- Key Performance Indicators (KPIs)
- Promotion distribution analysis
- Department-wise promotion rates
- Training score impact
- Performance rating analysis

### Tab 2: Demographics & Organization
- Gender and education analysis
- Regional performance comparison
- Recruitment channel effectiveness
- Service length patterns
- Department size analysis

### Tab 3: ML Predictions
- Real-time promotion prediction
- Input employee characteristics
- Get probability scores
- Model confidence metrics

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **F1-Score:** 0.257
- **Optimization:** GridSearchCV with 5-fold cross-validation
- **Handling Imbalance:** class_weight='balanced'
- **Best Parameters:**
  - criterion: 'gini'
  - max_depth: 10
  - random_state: 42

## 📈 Key Insights

1. **Previous Year Rating** is the strongest predictor (Rating 5.0 = 16% promotion rate)
2. **Education matters:** Master's degree holders have 11.5% promotion rate
3. **Training quality** is more important than quantity
4. **Department impact:** Technology leads with 12% promotion rate
5. **Referral advantage:** Referred employees have highest promotion rates (12.5%)

## 📂 Project Structure
```
employee-promotion-prediction/
│
├── employee_promotion_app.py          # Streamlit dashboard
├── employee_promotion_eda.ipynb       # EDA notebook
├── best_model.pkl                     # Trained ML model
├── scaled.pkl                         # Feature scaler
├── class_employee_promotion.csv       # Original dataset
├── Employee_Promotion_Cleaned.csv     # Cleaned dataset
├── Employee Promotion.jpg             # Dashboard image
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore file
└── README.md                          # Project documentation
```

## 👨‍💻 Author

**[Abdallah Ibrahim Mohammed Mustafa]**
- GitHub: [@abdallahebrahim785](https://github.com/abdallahebrahim785)
- LinkedIn: [Abdallah Ibrahim](https://www.linkedin.com/in/abdallah-ibrahim-4556792a5/)
- Email: abdallahebrahim785@gmail.com

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this project helpful, please give it a star!