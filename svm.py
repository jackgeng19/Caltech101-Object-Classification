# Import necessary libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from time import time

# Step 1: Reshape the data
# Reshape the images into a 2D array where each row is a flattened image
train_data_reshaped = train_data.reshape(train_data.shape[0], -1)
val_data_reshaped = val_data.reshape(val_data.shape[0], -1)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

# Step 2: Initialize the SVM Classifier
# Start with a basic SVM classifier with default parameters
svm_classifier = SVC()

# Step 3: Train the Model
# Train the SVM classifier with the training data
svm_classifier.fit(train_data_reshaped, train_labels)

# Step 4: Hyperparameter Tuning
def grid_search_svm(train_data, train_labels, val_data, val_labels):
    before = time()

    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        'kernel': ['linear', 'rbf', 'poly']  # Specifies the kernel type to be used in the algorithm
    }

    # Setup the grid search with cross-validation
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)

    # Perform the grid search on the training data
    grid_search.fit(train_data, train_labels)

    # Best parameters and best accuracy score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print('Best Parameters:', best_params)
    print('Best Validation Accuracy:', best_score)

    # Optionally: Evaluate on the validation set (not using cross-validation)
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print('Validation Accuracy:', val_accuracy)

    print("Time elapsed: ", time() - before)
    return best_model

# Call the grid search function with your data
best_svm_model = grid_search_svm(train_data_reshaped, train_labels, val_data_reshaped, val_labels)

# Step 5: Evaluate the Model
# Evaluate the best model's performance on the test data
predicted_labels = best_svm_model.predict(test_data_reshaped)
print("Accuracy on Test Data:", accuracy_score(test_labels, predicted_labels))
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))
