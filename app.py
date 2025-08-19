import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Global look & feel (elegant, crisp)
plt.style.use('default')
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.color": "#cbd5e1",
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

categorical_cols = ['membership_type', 'payment_history', 'attendance_record']
feature_names = [
    'age', 'membership_type', 'membership_length_months', 'avg_visits_per_month',
    'payment_delays', 'payment_history', 'attendance_record', 'satisfaction_score', 'membership_days'
]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    membership_type = request.form['membership_type']
    start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
    membership_length_months = int(request.form['membership_length_months'])
    avg_visits_per_month = int(request.form['avg_visits_per_month'])
    payment_delays = int(request.form['payment_delays'])
    payment_history = request.form['payment_history']
    attendance_record = request.form['attendance_record']
    satisfaction_score = int(request.form['satisfaction_score'])

    membership_days = (end_date - start_date).days
    if membership_days < 0:
        return "Error: End date must be after start date."

    input_data = pd.DataFrame({
        'age': [age],
        'membership_type': [membership_type],
        'membership_length_months': [membership_length_months],
        'avg_visits_per_month': [avg_visits_per_month],
        'payment_delays': [payment_delays],
        'payment_history': [payment_history],
        'attendance_record': [attendance_record],
        'satisfaction_score': [satisfaction_score],
        'membership_days': [membership_days]
    })[feature_names]

    input_data_encoded = input_data.copy()
    for col in categorical_cols:
        if col in label_encoders:
            input_data_encoded[col] = label_encoders[col].transform(input_data_encoded[col])

    prediction = model.predict(input_data_encoded)[0]
    proba = model.predict_proba(input_data_encoded)[0]
    churn_probability = proba[1] * 100
    risk_label = "High Risk of Churn" if prediction == 1 else "Low Risk of Churn"

    # LOGISTIC REGRESSION #######################################################################
    # Contributions
    coefficients = model.coef_[0]
    input_values = input_data_encoded.iloc[0].values
    contributions = input_values * coefficients
    #######################################################################
    
    # #RANDOM FOREST #######################################################################
    # # Contributions using feature importance (Random Forest does not have coefficients)
    # importances = model.feature_importances_
    # input_values = input_data_encoded.iloc[0].values

    # # Scale input by importance for pseudo "contributions"
    # contributions = input_values * importances
    #  #######################################################################
    
    

    if prediction == 1:
        sorted_indices = np.argsort(contributions)[::-1]
        reason_type = "Top reasons contributing to high churn risk:"
    else:
        sorted_indices = np.argsort(contributions)
        reason_type = "Top reasons supporting low churn risk:"

    top_3_indices = sorted_indices[:3]
    reasons = []
    top_features_labels = []
    top_features_values = []

    label_map = {
        'age': 'Age',
        'membership_type': 'Membership Type',
        'membership_length_months': 'Membership Length (Months)',
        'avg_visits_per_month': 'Avg Visits/Month',
        'payment_delays': 'Payment Delays',
        'payment_history': 'Payment History',
        'attendance_record': 'Attendance',
        'satisfaction_score': 'Satisfaction (1-10)',
        'membership_days': 'Membership Days'
    }

    for idx in top_3_indices:
        feature = feature_names[idx]
        value = input_data.iloc[0][feature]
        contrib = contributions[idx]
        top_features_labels.append(label_map.get(feature, feature))
        top_features_values.append(contrib)

        if feature == 'avg_visits_per_month':
            reasons.append("Low average visits ({} /month) suggests disengagement.".format(value) if contrib > 0
                           else "High average visits ({} /month) indicates strong engagement.".format(value))
        elif feature == 'payment_delays':
            reasons.append("Multiple payment delays ({}) increase churn likelihood.".format(value) if contrib > 0
                           else "Few payment delays ({}) support retention.".format(value))
        elif feature == 'satisfaction_score':
            reasons.append("Low satisfaction score ({}/10) points to dissatisfaction.".format(value) if contrib > 0
                           else "High satisfaction score ({}/10) boosts loyalty.".format(value))
        elif feature == 'membership_days':
            reasons.append("Short membership duration ({} days) often leads to quits.".format(value) if contrib > 0
                           else "Long membership duration ({} days) encourages staying.".format(value))
        elif feature == 'membership_length_months':
            reasons.append("Brief membership ({} months) raises risk.".format(value) if contrib > 0
                           else "Extended membership ({} months) lowers risk.".format(value))
        elif feature == 'age':
            reasons.append("Age ({}) correlates with higher churn in this group.".format(value) if contrib > 0
                           else "Age ({}) correlates with better retention.".format(value))
        elif feature == 'membership_type':
            reasons.append("{} membership type is associated with higher churn.".format(value) if contrib > 0
                           else "{} membership type promotes loyalty.".format(value))
        elif feature == 'payment_history':
            reasons.append("{} payment history signals potential issues.".format(value) if contrib > 0
                           else "{} payment history supports stability.".format(value))
        elif feature == 'attendance_record':
            reasons.append("{} attendance may indicate waning interest.".format(value) if contrib > 0
                           else "{} attendance shows commitment.".format(value))
        else:
            reasons.append("{} ({}) influences the prediction.".format(feature, value))

    if not os.path.exists('static'):
        os.makedirs('static')

    # Rich palette (accessible)
    emerald = "#22c55e"
    ruby = "#ef4444"
    sapphire = "#3b82f6"
    outline = "#0f172a"

    # ========== CHART 1: Churn Probability (reduced gap) ==========
    fig, ax = plt.subplots(figsize=(5.4, 3.3))

    values = proba * 100
    colors = [emerald, ruby]
    outline = "#374151"

    # Set manual x positions (reduce the gap between bars by lowering the difference)
    x_positions = [0, 0.35]  # second bar closer to first (default is ~0.8 to 1.0 apart)

    bars = ax.bar(x_positions, values, color=colors, width=0.3, edgecolor=outline, linewidth=0.8)

    # Set custom labels for x-axis ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Will Not Quit', 'Will Quit'])

    ax.set_ylabel('Probability (%)')
    ax.set_ylim(0, max(100, values.max() + 6))
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Annotate values
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color="#111827")

    plt.tight_layout(pad=1.0)
    churn_chart_path = 'static/churn_probability.png'
    plt.savefig(churn_chart_path, bbox_inches='tight')
    plt.close(fig)


    # ========== CHART 2: Top 3 Reasons (start at zero, slim bars, minimal gaps) ==========
    # Use absolute values for bar lengths (all positive)
    signed_values = np.array(top_features_values, dtype=float)
    plot_values = np.abs(signed_values)

    # Sort bars by magnitude descending
    order = np.argsort(plot_values)[::-1]
    plot_values = plot_values[order]
    signed_sorted = signed_values[order]
    labels_sorted = [top_features_labels[i] for i in order]

    fig2, ax2 = plt.subplots(figsize=(5.8, 3.2))
    y_pos = np.arange(len(labels_sorted))
    bar_height = 0.40  # can adjust thinner or thicker as needed

    # Colors indicating contribution sign: e.g., positive=blue, negative=red
    colors = ['#3b82f6' if val > 0 else "#3b82f6" for val in signed_sorted]

    bars2 = ax2.barh(y_pos, plot_values, height=bar_height,
                    color=colors, edgecolor='#0f172a', linewidth=0.8)

    ax2.set_yticks(y_pos, labels=labels_sorted)
    ax2.invert_yaxis()
    ax2.set_xlabel('Contribution (magnitude)')
    ax2.set_title('Top 3 Reasons for Prediction', pad=10)

    # Annotate with signed values (show sign in labels)
    for rect, signed in zip(bars2, signed_sorted):
        w = rect.get_width()
        ha = 'left' if w > 0 else 'right'
        offset = 5
        ax2.annotate(f"{signed:+.3f}",
                    xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords='offset points',
                    va='center',
                    ha=ha,
                    fontsize=9,
                    color="#111827")

    ax2.set_xlim(0, plot_values.max() * 1.1)
    ax2.xaxis.grid(True)
    ax2.yaxis.grid(False)

    plt.tight_layout()
    reasons_chart_path = 'static/top_features.png'
    plt.savefig(reasons_chart_path, bbox_inches='tight')
    plt.close(fig2)




    return render_template(
        'result.html',
        prediction="Will Quit" if prediction == 1 else "Will Not Quit",
        probability=round(churn_probability, 2),
        risk_label=risk_label,
        reason_type=reason_type,
        reasons=reasons,
        churn_chart=churn_chart_path,
        reasons_chart=reasons_chart_path
    )

if __name__ == '__main__':
    app.run(debug=True)
