from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import time

def train_and_evaluate(max_depth=None):
    # Load and prepare the dataset
    org_data = pd.read_csv("./salaries.csv")
    data = org_data.head(1000)

    learning_ratio = 0.9
    learning_size = int(len(data) * learning_ratio)

    # Divide the dataset
    learning_data = data.iloc[:learning_size]
    testing_data = data.iloc[learning_size:]

    # Separate features and target
    X_learning = learning_data.drop(columns='experience_level')
    Y_learning = learning_data['experience_level']
    X_testing = testing_data.drop(columns='experience_level')
    Y_testing = testing_data['experience_level']

    categoricalNames = ["work_year", "experience_level", "job_title", "company_location", "company_size"]

    # Encode features
    encoders = {}
    X_learning_encoded = pd.DataFrame()
    for column in X_learning.columns:
        if column in categoricalNames:
            encoder = LabelEncoder()
            X_learning_encoded[column] = encoder.fit_transform(X_learning[column])
            encoders[column] = encoder
        else:
            X_learning_encoded[column] = X_learning[column]

    # Encode target
    Y_encoder = LabelEncoder()
    Y_learning_encoded = Y_encoder.fit_transform(Y_learning)

    X_testing_encoded = pd.DataFrame()
    for column in X_testing.columns:
        if column in encoders:
            X_testing_encoded[column] = encoders[column].transform(X_testing[column])
        else:
            X_testing_encoded[column] = X_testing[column]

    Y_testing_encoded = Y_encoder.transform(Y_testing)

    # Initialize the Decision Tree model with specified max_depth
    decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)

    start_time = time.time()
    decision_tree.fit(X_learning_encoded, Y_learning_encoded)
    training_time = time.time() - start_time

    # Predict and evaluate
    Y_pred = decision_tree.predict(X_testing_encoded)
    accuracy = accuracy_score(Y_testing_encoded, Y_pred)
    conf_matrix = confusion_matrix(Y_testing_encoded, Y_pred)
    print(conf_matrix)

    print(f"Max Depth: {max_depth}, Accuracy: {accuracy}, Training Time: {training_time}s")

    # Visualize the decision tree
    plt.figure(figsize=(20,10))
    plot_tree(decision_tree, filled=True, rounded=True, class_names=Y_encoder.classes_, feature_names=X_learning_encoded.columns)
    plt.savefig(f"decision_tree_max_depth_{max_depth}.png")

    return accuracy, training_time

# Example of function call with max_depth=3
accuracy, training_time = train_and_evaluate(max_depth=10)

