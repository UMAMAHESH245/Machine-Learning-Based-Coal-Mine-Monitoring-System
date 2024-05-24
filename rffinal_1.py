import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset from Excel file
csv_file_path = "augmented_dataset_shuffle.csv"  # Replace with the actual path to your Excel file
df = pd.read_csv(csv_file_path)

# Map 'YES' and 'NO' to 1 and 0 for the 'Flame Detection' column
# df['FLAME DETECTED'] = df['FLAME DETECTED'].map({'Yes': 1, 'No': 0})

# Visualizations
# Histograms
df.hist(figsize=(10, 8))
plt.suptitle('Histograms of Features', y=1.02)
plt.show()

# Pairplot
sns.pairplot(df, hue='SAFETY CONDITION', diag_kind='kde')
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

# Countplot of 'Safety Condition'
plt.figure(figsize=(6, 4))
sns.countplot(x='SAFETY CONDITION', data=df)
plt.title('Distribution of Safety Condition')
plt.show()

# Split the dataset into features (X) and target variable (y)
X = df[['FLAME DETECTED', 'HUMIDITY', 'TEMPERATURE (Centigrade)', 'GAS LEVEL (PPM)']]
y = df['SAFETY CONDITION']

# Adjusting RandomForestClassifier hyperparameters for regularization
best_rf_model = RandomForestClassifier(random_state=42,
                                       n_estimators=50,
                                       max_depth=10,  # Adjust max_depth
                                       min_samples_split=5,  # Adjust min_samples_split
                                       min_samples_leaf=2)  # Adjust min_samples_leaf

# Use cross-validation to assess model performance
cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
print("\nCross-Validation Results:")
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f}")

# Train the model on the entire dataset
best_rf_model.fit(X, y)

# Print Feature Importances
feature_importances = pd.DataFrame(best_rf_model.feature_importances_,
                                   index=X.columns,
                                   columns=['Importance']).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Print Best Hyperparameters
print("\nBest Hyperparameters:")
print(f"Best Parameters: {best_rf_model.get_params()}")  # Print the parameters used in the final model

# Evaluate the model on the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
predictions = best_rf_model.predict(X_test)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

#Print Confusion Matrix
#print("\nConfusion Matrix:")
#print(confusion_matrix(y_test, predictions))

# Print Overall Accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nOverall Accuracy:")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file using joblib
joblib.dump(best_rf_model, "flame_model_regularized.joblib")