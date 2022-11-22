
# Import libraries
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

# Load dataset
numData = datasets.load_digits()

# Plotting + axis information for training data examples of 8x8 images
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, numData.images, numData.target):
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

# Flatten the image data, reducing file size
n_samples = len(numData.images)
data = numData.images.reshape((n_samples, -1))

# Declare classifier and adjustable gamma number
classifier = svm.SVC(gamma=0.001)

# Split data into testing and training images and labels
X_train, X_test, y_train, y_test = train_test_split(data, numData.target, test_size=0.5)

#  Image classification for training data
classifier.fit(X_train, y_train)

# Make a prediction based on training set on each
predicted = classifier.predict(X_test)

# Plotting numbers and predictions for 4 digits of the testing dataset (shown as example)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# Print accuracy scores
    print(f"Classification report: {classifier}:\n")
    print( f"{metrics.classification_report(y_test, predicted)}\n")


# Plot and display confusion matrix from comparison between testing label answers and prediction made by classifier.
display = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
display.figure_.suptitle("SVM Confusion Matrix")
print(f"Confusion matrix:\n{display.confusion_matrix}")

# Show plot on screen
plt.show()

