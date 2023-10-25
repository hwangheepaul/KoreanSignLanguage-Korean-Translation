import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load test data
file = np.genfromtxt('valid_add_data.csv', delimiter=',')
test_data = file[:, :-1].astype(np.float32)
test_labels = file[:, -1].astype(np.float32)

# Load training data
train_file = np.genfromtxt('train_add_data.csv', delimiter=',')
train_data = train_file[:, :-1].astype(np.float32)
train_labels = train_file[:, -1].astype(np.float32)

# Define K values
k_values = list(range(1, 100, 2))

accuracies = []

for k in k_values:
    # Train KNN model
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    # Test KNN model
    ret, results, _, _ = knn.findNearest(test_data, k)
    
    # Calculate accuracy
    accuracy = np.mean(results.flatten() == test_labels)
    accuracies.append(accuracy)

# Find the maximum accuracy and its corresponding K value
max_accuracy = max(accuracies)
max_k = k_values[accuracies.index(max_accuracy)]

# Plot accuracy vs K values
plt.plot(k_values, accuracies)
plt.scatter(max_k, max_accuracy, color='red', label=f'Max Accuracy: ({max_k}, {max_accuracy:.2f})')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K values')
plt.legend()
plt.show()