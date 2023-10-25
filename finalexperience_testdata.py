import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.set_printoptions(precision=4)
# Load test data
file = np.genfromtxt('test_add_data.csv', delimiter=',')
test_data = file[:, :-1].astype(np.float32)
test_labels = file[:, -1].astype(np.float32)

# Load training data
train_file = np.genfromtxt('gesture_train_fy_addtest.csv', delimiter=',')
train_data = train_file[:, :-1].astype(np.float32)
train_labels = train_file[:, -1].astype(np.float32)

# Define K value
k = 3

# Train KNN model
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Test KNN model
ret, results, _, _ = knn.findNearest(test_data, k)

# Calculate accuracy for unique labels
unique_labels = np.unique(train_labels)
accuracies = []

for label in unique_labels:
    accuracy = np.mean(results[test_labels == label].flatten() == label)
    accuracies.append(accuracy)
mean_accuracy = np.nanmean(accuracies)


print("테스트 데이터의 정확도:", accuracy)
print("테스트 데이터의 평균 정확도:", mean_accuracy)

# Find data points with accuracy < 0.9
low_accuracy_indices = np.where(np.array(accuracies) < 0.7)[0]
low_accuracy_labels = unique_labels[low_accuracy_indices]
low_accuracy_values = np.array(accuracies)[low_accuracy_indices]

# Plot accuracy vs unique labels
plt.plot(unique_labels, accuracies)
plt.scatter(low_accuracy_labels, low_accuracy_values, c='blue', alpha=0.5)
plt.axhline(y=mean_accuracy, color='red', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.3f}')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.xlabel('hand_type index')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Index')
plt.legend()
plt.show()