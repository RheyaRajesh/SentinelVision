import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Simulate dataset (replace with real: extract from YOLO on labeled alerts)
# Features: [confidence, normalized_bbox_size, duration, flow_magnitude]
# Labels: 1=true, 0=false
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 4)
# Simulate labels: High conf/large size = true
y = ((X[:, 0] > 0.6) & (X[:, 1] > 0.05) & (X[:, 2] > 2.0)).astype(int)

# Real extraction example: Use utils.feature_extractor.py on validation videos
# df = pd.read_csv('validation_features.csv')  # Assume CSV with features and labels
# X, y = df[['conf', 'bbox_norm', 'dur', 'flow']].values, df['true'].values

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Individual models
svm = SVC(kernel='rbf', probability=True)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)

# Train
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Ensemble
ensemble = VotingClassifier(
    estimators=[('svm', svm), ('dt', dt), ('rf', rf), ('lr', lr)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Evaluate
preds = ensemble.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# Save
joblib.dump(ensemble, 'models/alert_classifier.pkl')
joblib.dump(scaler, 'models/scaler.pkl')  # For inference scaling
print("Models saved.")