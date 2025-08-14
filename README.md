# 🏋️‍♂️ Gym Membership Churn Prediction App  

A Flask-based web application that predicts whether a gym member is likely to **churn** (cancel their membership) based on details like age, membership type, attendance, payment history, and satisfaction score. The app uses a trained **machine learning model** to provide predictions and probability scores along with **top 3 reasons for churn**.  

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://your-live-demo-link.com)

## 🚀 Features  
- **Interactive Web Form** to input member details  
- **Automatic Membership Length Calculation** from start and end dates  
- **ML Prediction** (Churn or Not Churn)  
- **Probability of Churn** (in %)  
- **Top 3 Reasons for Predicted Churn** for insights  
- Elegant **Olive Green UI** for better user experience  

## 🛠 Tech Stack  
- **Backend:** Python, Flask  
- **Frontend:** HTML5, CSS3, JavaScript  
- **ML Libraries:** scikit-learn, pandas, numpy  
- **Visualization:** Matplotlib  

## 📂 Project Structure  
```
├── app.py                # Flask backend  
├── model.pkl             # Trained ML model  
├── label_encoders.pkl    # Encoders for categorical features  
├── templates/  
│   ├── index.html        # Input form  
│   └── result.html       # Prediction output page  
├── requirements.txt      # Python dependencies  
└── README.md             # Project documentation  
```

## 📦 Installation  
1. Clone the repository  
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
2. Install dependencies  
```bash
pip install -r requirements.txt
```
3. Run the app  
```bash
python app.py
```
4. Open in browser:  
```
http://127.0.0.1:5000
```

## 🌐 Deployment  
You can deploy this app on:  
- [Render](https://render.com/)  
- [Railway](https://railway.app/)  
- [Heroku](https://www.heroku.com/)  

## 📊 Example Prediction  
**Input:**  
- Age: 23  
- Membership Type: Basic  
- Membership Length: 3 months  
- Avg Visits/Month: 20  
- Payment Delays: 2  
- Payment History: Mixed  
- Attendance: Regular  
- Satisfaction: 5/10  

**Output:**  
- Prediction: **Churn**  
- Probability: **99.97%**  
- Reasons: Low satisfaction, Payment delays, Short membership period  
