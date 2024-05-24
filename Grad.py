import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset from CSV file
csv_file_path = "my_flame_dataset_with_labels.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Map 'YES' and 'NO' to 1 and 0 for the 'Flame Detection' column
df['Flame Detection'] = df['Flame Detection'].map({'YES': 1, 'NO': 0})

# Visualizations
# Histograms
df.hist(figsize=(10, 8))
plt.suptitle('Histograms of Features', y=1.02)
plt.show()

# Pairplot
sns.pairplot(df, hue='Safety Condition', diag_kind='kde')
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

# Countplot of 'Safety Condition'
plt.figure(figsize=(6, 4))
sns.countplot(x='Safety Condition', data=df)
plt.title('Distribution of Safety Condition')
plt.show()

# Split the dataset into features (X) and target variable (y)
X = df[['Flame Detection', 'Humidity (%)', 'Temperature (°C)', 'Gas Level (PPM)']]
y = df['Safety Condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

# Fit the model to the training data
gb_model.fit(X_train, y_train)

# Make predictions on the test data
predictions_gb = gb_model.predict(X_test)

# Print the classification report for Gradient Boosting
print("Gradient Boosting Classifier:")
print(classification_report(y_test, predictions_gb))

# Evaluate the Gradient Boosting model
accuracy_gb = accuracy_score(y_test, predictions_gb)
conf_matrix_gb = confusion_matrix(y_test, predictions_gb)
class_report_gb = classification_report(y_test, predictions_gb)

# Print the results for Gradient Boosting
print(f'Accuracy (Gradient Boosting): {accuracy_gb * 100:.2f}%')
print('\nConfusion Matrix (Gradient Boosting):')
print(conf_matrix_gb)
print('\nClassification Report (Gradient Boosting):')
print(class_report_gb)

# Save the trained Gradient Boosting model to a file using joblib
joblib.dump(gb_model, "gradient_boosting_flame_model.joblib")
