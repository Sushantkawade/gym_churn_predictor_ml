import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------
# 1. Load dataset
# -------------------
df = pd.read_csv("data/gym_churn_100entries.csv")

# -------------------
# 2. Convert dates to datetime
# -------------------
df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

# -------------------
# 3. Create membership duration in days
# -------------------
df["membership_days"] = (df["end_date"] - df["start_date"]).dt.days

# Drop original date columns (not needed anymore)
df = df.drop(["start_date", "end_date"], axis=1)

# -------------------
# 4. Separate features & target
# -------------------
X = df.drop("quit", axis=1)
y = df["quit"]

# -------------------
# 5. Encode categorical variables
# -------------------
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -------------------
# 6. Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------
# 7. Train Logistic Regression model
# -------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------
# 8. Predictions & probabilities
# -------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of quitting

# -------------------
# 9. Show some results
# -------------------
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Quit_Probability": y_pred_proba
})
print(results.head())

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------
# 10. Save model & encoders
# -------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("\n✅ Model trained and saved successfully.")