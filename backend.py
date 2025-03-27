import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from collections import Counter
from flask import Flask, request, jsonify

# Load dataset
file_path = r"D:\pro-1\nivas.1\datasets.csv"
df = pd.read_csv(file_path)

# Check required columns
if 'url' not in df.columns or 'target' not in df.columns:
    raise ValueError("Required columns missing. Ensure 'url' and 'target' are present.")

if set(df['target'].unique()) - {0, 1}:
    raise ValueError("Unexpected target values detected.")

print("Class Distribution:\n", df['target'].value_counts())

# Extract features
X = df['url']
y = df['target']

# Feature extraction function
def extract_features(url):
    url = str(url).lower()
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    extracted = tldextract.extract(url)
    
    suspicious_keywords = ['secure', 'account', 'update', 'login', 'bank', 'verify', 'password']
    keyword_count = sum(1 for word in suspicious_keywords if word in url)
    
    features = [
        len(url),  
        len(re.findall(r'[^\w\s]', url)),  
        0 if 'https' in url else 1,  
        domain.count('.'),  
        len(domain),  
        len(re.findall(r'\d', url)),  
        1 if '@' in url else 0,  
        1 if '-' in domain else 0,  
        len(parsed_url.query.split('&')) if parsed_url.query else 0,  
        1 if 'www' in domain else 0,  
        len(extracted.subdomain),  
        extracted.subdomain.count('.') + 1 if extracted.subdomain else 0,  
        len(extracted.suffix),  
        keyword_count,  
        len(re.findall(r'\d', url)) / len(url)  
    ]
    return features

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Extract features
feature_names = ['url_length', 'special_chars', 'https', 'dot_count', 'domain_length',
                 'num_count', 'has_at', 'has_hyphen', 'query_count', 'has_www',
                 'subdomain_length', 'num_subdomains', 'tld_length', 'keyword_count', 'digit_ratio']

X_train_features = pd.DataFrame([extract_features(url) for url in X_train], columns=feature_names)
X_test_features = pd.DataFrame([extract_features(url) for url in X_test], columns=feature_names)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

# Hyperparameters
param_grids = {
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Random Forest": {'n_estimators': [50, 100], 'max_depth': [5, 10]},
    "SVM": {'C': [0.1, 1], 'gamma': ['scale']},
    "XGBoost": {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]},
    "Neural Network": {'hidden_layer_sizes': [(128, 64), (64, 32)], 'alpha': [0.0001, 0.001]}
}

# Train and tune models
for name, model in models.items():
    print(f"\nTuning {name}...")
    random_search = RandomizedSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_iter=3, n_jobs=-1, random_state=42)
    random_search.fit(X_train_features, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(X_test_features)
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['safe', 'phishing']))
    
    # Save best model
    model_filename = f"phishing_detection_model_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\nBest model saved as {model_filename}")

    # Cross-validation
    scores = cross_val_score(best_model, X_train_features, y_train, cv=3)
    print(f"{name} Cross-validation scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}")

# Majority Voting Function
def majority_voting(models, test_features):
    predictions = []
    for name in models.keys():
        model = joblib.load(f"phishing_detection_model_{name.lower().replace(' ', '_')}.pkl")
        prediction = model.predict(test_features)[0]  # Ensure test_features is a DataFrame
        predictions.append(prediction)
    
    # Count the votes
    counter = Counter(predictions)
    most_common = counter.most_common()
    
    # If there is a tie, use Random Forest as the tiebreaker
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tiebreaker_model = joblib.load("phishing_detection_model_random_forest.pkl")
        tiebreaker_prediction = tiebreaker_model.predict(test_features)[0]
        return tiebreaker_prediction
    else:
        return most_common[0][0]

# Example: Replace the placeholder URL with your deployed backend URL
backend_url = "http://127.0.0.1:5000"  # Replace with your deployed backend URL if hosting remotely

# Example usage of the backend URL
print(f"Backend is running at: {backend_url}")

# Test with an example URL
test_url = "https://bank6382-verify.com"
test_features = extract_features(test_url)
test_df = pd.DataFrame([test_features], columns=feature_names)  # Ensure test_features is wrapped in a DataFrame

# Ensure the test DataFrame matches the expected format
if test_df.shape[1] != len(feature_names):
    raise ValueError("Feature extraction mismatch. Ensure the feature extraction function outputs the correct number of features.")

final_prediction = majority_voting(models, test_df)
print(f"\nFinal Prediction (Majority Vote) for {test_url}: {'Phishing' if final_prediction == 1 else 'Safe'}")

# Initialize Flask app
app = Flask(__name__)

# Load models
models = {
    "Logistic Regression": joblib.load("phishing_detection_model_logistic_regression.pkl"),
    "Random Forest": joblib.load("phishing_detection_model_random_forest.pkl"),
    "SVM": joblib.load("phishing_detection_model_svm.pkl"),
    "XGBoost": joblib.load("phishing_detection_model_xgboost.pkl"),
    "Neural Network": joblib.load("phishing_detection_model_neural_network.pkl")
}

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'url' not in data:
        return jsonify({"error": "Missing 'url' in request data"}), 400

    url = data['url']
    features = extract_features(url)
    test_df = pd.DataFrame([features], columns=feature_names)

    if test_df.shape[1] != len(feature_names):
        return jsonify({"error": "Feature extraction mismatch"}), 500

    prediction = majority_voting(models, test_df)
    result = "Phishing" if prediction == 1 else "Safe"
    return jsonify({"url": url, "prediction": result})

# Deploy with Renter
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)