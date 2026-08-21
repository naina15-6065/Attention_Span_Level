from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# Load preprocessing objects
encoder = joblib.load("model/ordinal_encoder.pkl")
le = joblib.load("model/label_encoder.pkl")
selector = joblib.load("model/selector.pkl")
scaler = joblib.load("model/scaler.pkl")

et = joblib.load("model/extra_trees.pkl")

# Load config
with open("model/hybrid_config.pkl", "rb") as f:
    config = pickle.load(f)

ALL_COLUMNS = config["all_columns"]

# Load feature importance
feature_importance_df = pd.read_csv("model/feature_importance.csv")
top_features = feature_importance_df.head(5).to_dict(orient="records")

print("Label classes:", le.classes_)


def build_explanations(raw_input):
    explanations = []

    if raw_input.get("Screen_Time") in ["8 hours or more", "6-7 hours"]:
        explanations.append("Higher screen time may reduce sustained attention.")

    if raw_input.get("Sleep_Hours") in ["7-8 hours", "9 hours or more"]:
        explanations.append("Adequate sleep may support better attention and focus.")
    elif raw_input.get("Sleep_Hours") == "Less than 5 hours":
        explanations.append("Insufficient sleep may negatively affect attention span.")

    if raw_input.get("Sleep_Quality") in ["4", "5"]:
        explanations.append("Good sleep quality is positively associated with concentration.")
    elif raw_input.get("Sleep_Quality") in ["1", "2"]:
        explanations.append("Poor sleep quality may reduce mental alertness.")

    if raw_input.get("Physical_Activity") in ["3-5 days", "more than 5 days"]:
        explanations.append("Regular physical activity may improve cognitive performance.")

    if raw_input.get("Stress_Level") in ["4", "5"]:
        explanations.append("Higher stress may make it more difficult to maintain focus.")

    if raw_input.get("Multitasking") in ["Often", "Always"]:
        explanations.append("Frequent multitasking may reduce the ability to focus on a single task.")

    if raw_input.get("Long_Form_Content") in ["Sometimes", "Often"]:
        explanations.append("Engaging with long-form content may support sustained attention.")

    if not explanations:
        explanations.append("The prediction is based on the combined influence of the provided lifestyle and behavioral factors.")

    return explanations


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_input = request.form.to_dict()
        print("Raw input:", raw_input)

        df = pd.DataFrame([raw_input])

        # numeric fields
        df["Sleep_Quality"] = df["Sleep_Quality"].astype(int)
        df["Stress_Level"] = df["Stress_Level"].astype(int)

        # ordinal encoding
        df[config["ordinal_cols"]] = encoder.transform(df[config["ordinal_cols"]])

        # one-hot encoding
        df = pd.get_dummies(df)

        # align columns
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df = df[ALL_COLUMNS]

        print("Aligned input row:")
        print(df.iloc[0].to_dict())

        # feature selection
        X_selected = selector.transform(df)

        # scaling
        X_scaled = scaler.transform(X_selected)

        # Extra Trees prediction only
        et_proba = et.predict_proba(X_scaled)
        probs = et_proba[0]
        classes = le.classes_

        pred_class = np.argmax(probs)
        prediction = classes[pred_class]

        print("ET probabilities:", et_proba)
        print("Predicted index:", pred_class)
        print("Predicted label:", prediction)

        probabilities = {
            classes[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(classes))
        }

        explanations = build_explanations(raw_input)

        return render_template(
            "index.html",
            prediction=prediction,
            probabilities=probabilities,
            top_features=top_features,
            explanations=explanations,
            form_data=raw_input
        )

    except Exception as e:
        print("ERROR:", e)
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)