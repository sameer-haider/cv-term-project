import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Constants
TRAIN_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/Train"
TEST_PATH = "/ProjData/Train/"

RESIZED_BEDROOM_PATH_200 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/bedroom/200"
RESIZED_BEDROOM_PATH_50 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/bedroom/50"
BEDROOM_PATH_GRAY = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/bedroom/gray"

RESIZED_COAST_PATH_200 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/Coast/200"
RESIZED_COAST_PATH_50 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/Coast/50"
COAST_PATH_GRAY = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/Coast/gray"

RESIZED_FOREST_PATH_200 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/Forest/200"
RESIZED_FOREST_PATH_50 = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/Forest/50"
FOREST_PATH_GRAY = "/Users/sanjith/Desktop/cv-term-project/ProjData/resized/Forest/gray"

SIFT_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/sift"
HISTOGRAM_PATH = "/Users/sanjith/Desktop/cv-term-project/ProjData/histogram"

img_features = []
target_lables = []

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


## 2 and 3
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


## 4
def train_classifier_nn():
    features = []
    target_labels = []
    class_names = ['bedroom', 'Coast', 'Forest']
    folder_path = '/Users/sanjith/Desktop/cv-term-project/ProjData/resized'
    # Loop through each class folder and read all images inside
    for i, class_name in enumerate(class_names):
        images_path = os.path.join(folder_path, class_name, '50')
        for img_name in os.listdir(images_path):
            img_path = os.path.join(images_path, img_name)
            # Load image and resize it to 50x50 pixels
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Flatten the image pixel values to a 1D array and add to features list
            img_features = image.flatten()
            features.append(img_features)
            # Add the target label for the class to the target_labels list
            target_labels.append(i)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, target_labels)


    # Testing data preparation
    test_data = []
    test_labels = []

    for label, folder_name in enumerate(['bedroom', 'Coast', 'Forest']):
        folder_path = os.path.join('/Users/sanjith/Desktop/cv-term-project/ProjData/Train', folder_name)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))
            test_data.append(img.flatten())
            test_labels.append(label)

    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels)

    # Nearest Neighbor classifier testing
    result = knn.predict(test_data)

    # Compute accuracy
    correct = np.count_nonzero(result == test_labels)
    accuracy = correct * 100.0 / len(test_labels)
    print('Accuracy: {:.2f}%'.format(accuracy))

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
        if os.path.isdir(folder_path):
        # Loop through the files in the current folder
            for filename in os.listdir(folder_path):
                # Perform operations on the file
                img_path = os.path.join(folder_path, filename)

                gray, resized_200, resized_50 = preprocess_image(img_path)
                new_filename_gray = 'gray_' + filename
                new_filename_200 = 'resized_200_' + filename
                new_filename_50 = 'resized_50_' + filename

                dir_200 = ""
                dir_50 = ""
                dir_gray = ""

                if folder_name == 'bedroom':
                    dir_200 = RESIZED_BEDROOM_PATH_200
                    dir_50 = RESIZED_BEDROOM_PATH_50
                    dir_gray = BEDROOM_PATH_GRAY
                elif folder_name == 'Coast':
                    dir_200 = RESIZED_COAST_PATH_200
                    dir_50 = RESIZED_COAST_PATH_50
                    dir_gray = COAST_PATH_GRAY
                elif folder_name == 'Forest':
                    dir_200 = RESIZED_FOREST_PATH_200
                    dir_50 = RESIZED_FOREST_PATH_50
                    dir_gray = FOREST_PATH_GRAY
                
                new_img_path_200 = os.path.join(dir_200, new_filename_200)
                new_img_path_50 = os.path.join(dir_50, new_filename_50)
                new_img_path_gray = os.path.join(dir_gray, new_filename_gray)
                cv2.imwrite(new_img_path_gray, gray)
                cv2.imwrite(new_img_path_200, resized_200)
                cv2.imwrite(new_img_path_50, resized_50)
    """

    # Train classifiers with different representations and methods
    # ...
    train_classifier_nn()

    # Test the classifiers using test images and report results
    # ...
    
    return


if __name__ == "__main__":
    main()
