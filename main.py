import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Constants
TRAIN_PATH = "path/to/train/dataset"
TEST_PATH = "path/to/test/dataset"
RESIZED_PATH_200 = "path/to/resized/200"
RESIZED_PATH_50 = "path/to/resized/50"
GRAYSCALE_PATH = "path/to/grayscale"
SIFT_PATH = "path/to/sift"
HISTOGRAM_PATH = "path/to/histogram"


## 1
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust brightness if necessary
    mean_brightness = np.mean(gray)
    if mean_brightness < 0.4 * 255:
        gray = cv2.add(gray, 40)
    elif mean_brightness > 0.6 * 255:
        gray = cv2.subtract(gray, 40)

    # Resize to two different sizes and save them
    resized_200 = cv2.resize(gray, (200, 200))
    resized_50 = cv2.resize(gray, (50, 50))

    return gray, resized_200, resized_50


## 2
def extract_sift_features(gray_image):
    # Initialize SIFT feature extractor
    sift = cv2.SIFT_create()

    # Compute SIFT features
    _, descriptors = sift.detectAndCompute(gray_image, None)

    return descriptors


## 3
def extract_histogram_features(gray_image, bins=16):
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [bins], [0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist)

    return hist.flatten()


## 4
def train_classifier(training_data, training_labels, classifier_type="knn"):
    if classifier_type == "knn":
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier_type == "svm":
        classifier = LinearSVC()

    classifier.fit(training_data, training_labels)
    return classifier


## 5
def test_classifier(classifier, test_data, test_labels):
    predicted_labels = classifier.predict(test_data)

    accuracy = accuracy_score(test_labels, predicted_labels)
    cm = confusion_matrix(test_labels, predicted_labels)

    return accuracy, cm


# Main function
def main():
    # Load training and test datasets, preprocess images, and extract features
    # ...

    # Train classifiers with different representations and methods
    # ...

    # Test the classifiers using test images and report results
    # ...

    # Write a report and analyze the results
    # ...

    return


if __name__ == "__main__":
    main()
