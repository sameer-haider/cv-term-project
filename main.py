import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Constants
TRAIN_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/Train"
TEST_PATH = "/ProjData/Train/"
RESIZED_PATH_200 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/200"
RESIZED_PATH_50 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/50"
GRAYSCALE_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/grayscale"
SIFT_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/sift"
HISTOGRAM_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/histogram"


## 1
def preprocess_image(image_path):
    print(image_path)
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
def extract_sift_features(gray_image, file_name):
    # Initialize SIFT feature extractor
    sift = cv2.SIFT_create()
    # Compute SIFT features
    kp, des = sift.detectAndCompute(gray_image, None)

    # Save the SIFT features to a file
    # Save the descriptors to a file
    desc_path_sift = os.path.join(SIFT_PATH, file_name.replace('.jpg', '.txt'))
    np.savetxt(desc_path_sift, des)

    # histogram
    # Calculate a histogram of the SIFT descriptors
    hist = cv2.calcHist([des], [0], None, [256], [0, 256])

    # Normalize the histogram to values between 0 and 1
    hist = cv2.normalize(hist, hist)
    desc_path_hist = os.path.join(HISTOGRAM_PATH, file_name.replace('.jpg', '.txt'))
    np.savetxt(desc_path_hist, hist)



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
    """
    parent_folder = '/Users/sanjith/Desktop/cv-term-project/ProjData/Train'
    folder_name_list = ['bedroom', 'Coast', 'Forest']
    for folder_name in os.listdir(parent_folder):
        # Check if the current item in the folder is a folder
        if folder_name in folder_name_list:
            folder_path = os.path.join(parent_folder, folder_name)
        else:
            continue
        print(f"folder_path: {folder_path}")
        if os.path.isdir(folder_path):
        # Loop through the files in the current folder
            for filename in os.listdir(folder_path):
                # Perform operations on the file
                print(f"filename: {filename}")
                img_path = os.path.join(folder_path, filename)

                gray, resized_200, resized_50 = preprocess_image(img_path)
                new_filename_gray = 'gray_' + filename
                new_filename_200 = 'resized_200_' + filename
                new_filename_50 = 'resized_50_' + filename
                new_img_path_200 = os.path.join(RESIZED_PATH_200, new_filename_200)
                new_img_path_50 = os.path.join(RESIZED_PATH_50, new_filename_50)
                new_img_path_gray = os.path.join(GRAYSCALE_PATH, new_filename_gray)
                cv2.imwrite(new_img_path_gray, gray)
                cv2.imwrite(new_img_path_200, resized_200)
                cv2.imwrite(new_img_path_50, resized_50)
    """

    # Train classifiers with different representations and methods
    # ...

    # Test the classifiers using test images and report results
    # ...
    for file_name in os.listdir(GRAYSCALE_PATH):
        image_path = os.path.join(GRAYSCALE_PATH, file_name)
        gray_image = cv2.imread(image_path)
        img = extract_sift_features(gray_image, file_name)
        pass

    return


if __name__ == "__main__":
    main()
