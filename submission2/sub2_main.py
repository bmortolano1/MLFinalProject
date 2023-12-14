from sklearn.svm import NuSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import sub2_data_processor as dp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
    # Process test and train data
    [features_train, labels_train] = dp.get_train_data()
    features_test = dp.get_test_data()

    # Nu-Support Vector Classification
    clf = NuSVC(nu=0.45, probability=True)

    clf.fit(features_train, labels_train)

    # Optional: Calculate error of train data
    labels_train_pred = clf.predict(features_train)
    accuracy = accuracy_score(labels_train, labels_train_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Test the classifier on the test data
    # labels_test_pred = clf.predict_proba(features_test)
    # labels_test_pred = [x[1] for x in labels_test_pred]
    # dp.output_test_file(labels_test_pred, '../out.csv')

    clf1 = LogisticRegression(multi_class='auto', random_state=1, max_iter=1000, C=0.001)
    clf2 = RandomForestClassifier(n_estimators=250, random_state=42, max_depth=100, min_samples_leaf=5, max_features=11, max_samples=0.25)
    clf3 = GaussianNB(var_smoothing=1e-5)
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

    eclf1 = eclf1.fit(features_train, labels_train)

    # Optional: Calculate error of train data
    labels_train_pred = eclf1.predict(features_train)
    accuracy = accuracy_score(labels_train, labels_train_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Test the classifier on the test data
    # labels_test_pred = eclf1.predict_proba(features_test)
    # labels_test_pred = [x[1] for x in labels_test_pred]
    # dp.output_test_file(labels_test_pred, '../out.csv')

    nclf1 = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100,100), early_stopping=True, max_iter=10000)
    nclf1 = nclf1.fit(features_train, labels_train)

    # Optional: Calculate error of train data
    labels_train_pred = nclf1.predict(features_train)
    accuracy = accuracy_score(labels_train, labels_train_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Test the classifier on the test data
    labels_test_pred = nclf1.predict_proba(features_test)
    labels_test_pred = [x[1] for x in labels_test_pred]
    dp.output_test_file(labels_test_pred, '../out.csv')