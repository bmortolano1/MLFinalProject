from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sub1_data_processor as dp

# Process test and train data

[features_train, labels_train] = dp.get_train_data()
features_test = dp.get_train_data()

# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees (n_estimators)

# Train the classifier on the training data
clf.fit(features_train, labels_train)

# Optional: Calculate error of train data
labels_train_pred = clf.predict(features_test)
accuracy = accuracy_score(labels_train, labels_train_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test the classifier on the test data
labels_test_pred = clf.predict(features_test)