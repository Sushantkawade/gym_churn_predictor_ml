import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Global look & feel
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

reason_templates = {
    'avg_visits_per_month': lambda v, c: f"Low average visits ({v} /month) suggests disengagement." if c > 0 else f"High average visits ({v} /month) indicates strong engagement.",
    'payment_delays': lambda v, c: f"Multiple payment delays ({v}) increase churn likelihood." if c > 0 else f"Few payment delays ({v}) support retention.",
    'satisfaction_score': lambda v, c: f"Low satisfaction score ({v}/10) points to dissatisfaction." if c > 0 else f"High satisfaction score ({v}/10) boosts loyalty.",
    'membership_days': lambda v, c: f"Short membership duration ({v} days) often leads to quits." if c > 0 else f"Long membership duration ({v} days) encourages staying.",
    'membership_length_months': lambda v, c: f"Brief membership ({v} months) raises risk." if c > 0 else f"Extended membership ({v} months) lowers risk.",
    'age': lambda v, c: f"Age ({v}) correlates with higher churn in this group." if c > 0 else f"Age ({v}) correlates with better retention.",
    'membership_type': lambda v, c: f"{v} membership type is associated with higher churn." if c > 0 else f"{v} membership type promotes loyalty.",
    'payment_history': lambda v, c: f"{v} payment history signals potential issues." if c > 0 else f"{v} payment history supports stability.",
    'attendance_record': lambda v, c: f"{v} attendance may indicate waning interest." if c > 0 else f"{v} attendance shows commitment."
}

def get_form_values(request):
    fields = [
        ('age', int), ('membership_type', str),
        ('membership_length_months', int), ('avg_visits_per_month', int),
        ('payment_delays', int), ('payment_history', str),
        ('attendance_record', str), ('satisfaction_score', int)
    ]
    values = {name: cast(request.form[name]) for name, cast in fields}
    start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
    values['membership_days'] = (end_date - start_date).days
    return values, start_date, end_date

def encode_input(input_data, encoders, cat_cols):
    for col in cat_cols:
        input_data[col] = encoders[col].transform(input_data[col])
    return input_data

def get_reasons(sorted_indices, feature_names, input_data, contributions):
    return [
        reason_templates.get(
            feature_names[idx],
            lambda v, c: f"{feature_names[idx]} ({v}) influences the prediction."
        )(input_data.iloc[0][feature_names[idx]], contributions[idx])
        for idx in sorted_indices[:3]
    ]

def generate_bar_chart(path, values, colors, x_labels, y_label):
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x_positions = np.arange(len(values))
    bars = ax.bar(x_positions, values, color=colors, width=0.3, edgecolor="#374151", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, max(100, max(values) + 6))
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=9, color="#111827")
    plt.tight_layout(pad=1.0)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

def generate_horizontal_chart(path, values, labels, signed_values):
    fig2, ax2 = plt.subplots(figsize=(5.8, 3.2))
    y_pos = np.arange(len(labels))
    bar_height = 0.40
    bars2 = ax2.barh(y_pos, values, height=bar_height, color="#3b82f6", edgecolor='#0f172a', linewidth=0.8)
    ax2.set_yticks(y_pos, labels=labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Contribution (magnitude)')
    ax2.set_title('Top 3 Reasons for Prediction', pad=10)
    for rect, signed in zip(bars2, signed_values):
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
    ax2.set_xlim(0, max(values) * 1.1)
    ax2.xaxis.grid(True)
    ax2.yaxis.grid(False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig2)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_values, start_date, end_date = get_form_values(request)
    if form_values['membership_days'] < 0:
        return "Error: End date must be after start date."
    input_data = pd.DataFrame([form_values])[feature_names]

    input_data_encoded = input_data.copy()
    input_data_encoded = encode_input(input_data_encoded, label_encoders, categorical_cols)

    prediction = model.predict(input_data_encoded)[0]
    proba = model.predict_proba(input_data_encoded)[0]
    churn_probability = proba[1] * 100
    risk_label = "High Risk of Churn" if prediction == 1 else "Low Risk of Churn"

    # Coefficients for logistic regression 
    coefficients = model.coef_[0]
    input_values = input_data_encoded.iloc[0].values
    contributions = input_values * coefficients

    sorted_indices = np.argsort(contributions)[::-1] if prediction == 1 else np.argsort(contributions)
    reason_type = "Top reasons contributing to high churn risk:" if prediction == 1 else "Top reasons supporting low churn risk:"

    top_features_labels = [label_map.get(feature_names[idx], feature_names[idx]) for idx in sorted_indices[:3]]
    top_features_values = [contributions[idx] for idx in sorted_indices[:3]]
    reasons = get_reasons(sorted_indices, feature_names, input_data, contributions)

    if not os.path.exists('static'):
        os.makedirs('static')

    # Chart 1: Churn Probability
    churn_chart_path = 'static/churn_probability.png'
    generate_bar_chart(
        churn_chart_path,
        proba * 100,
        ["#22c55e", "#ef4444"],
        ['Will Not Quit', 'Will Quit'],
        'Probability (%)'
    )

    # Chart 2: Top 3 Reasons
    signed_values = np.array(top_features_values, dtype=float)
    plot_values = np.abs(signed_values)
    order = np.argsort(plot_values)[::-1]
    plot_values = plot_values[order]
    signed_sorted = signed_values[order]
    labels_sorted = [top_features_labels[i] for i in order]

    reasons_chart_path = 'static/top_features.png'
    generate_horizontal_chart(reasons_chart_path, plot_values, labels_sorted, signed_sorted)

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
