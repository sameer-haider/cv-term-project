import os
import cv2
import numpy as np

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
def extract_sift_features(img_path, file_name, folder_name):

    gray_img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT feature extractor
    sift = cv2.SIFT_create()
    # Compute SIFT features
    kp, des = sift.detectAndCompute(gray_img, None)

    parent_folder = '/Users/sanjith/Desktop/cv-term-project/ProjData/resized'

    # Save the SIFT features to a file
    # Save the descriptors to a file
    desc_path_sift = os.path.join(parent_folder, folder_name, 'sift' ,file_name.replace('.jpg', '.txt'))
    print(desc_path_sift)
    if des is not None:
        np.save(desc_path_sift, des)


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

    # Convert the training data and labels to numpy arrays
    train_data = np.array(features, dtype=np.float32)
    train_labels = np.array(target_labels, dtype=np.float32)

    # Create the kNN classifier with k=1
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(1)

    # Train the kNN classifier with the training data and labels
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

def train_classifier_sift_nn():
    features = []
    target_labels = []
    class_names = ['bedroom', 'Coast', 'Forest']
    folder_path = '/Users/sanjith/Desktop/cv-term-project/ProjData/resized'
    # Loop through each class folder and read all images inside
    for i, class_name in enumerate(class_names):
        images_path = os.path.join(folder_path, class_name,'sift')
        print(images_path)
        for img_name in os.listdir(images_path):
            print(img_name)
            if not img_name.endswith('.npy'):
                continue
            img_path = os.path.join(images_path, img_name)
            des = np.load(img_path, allow_pickle=True)
            features.append(des.flatten())
            # Append the label to the training labels list
            target_labels.append(class_name)

    # Convert the training data and labels to numpy arrays
    train_data = np.array(features, dtype=np.float32)
    train_labels = np.array(target_labels, dtype=np.float32)

    # Create the kNN classifier with k=1
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(1)

    # Train the kNN classifier with the training data and labels
    print(train_data)
    print(train_labels)
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)




def train_and_test_sift_NN():
    img_features_sift_train = []
    target_lables_sift_train = []
    descriptors = []
    #desc_label = np.empty((0, 128))

    actual_labels = []

    train_path = '/Users/sanjith/Desktop/cv-term-project/ProjData/resized'
    train_data = []
    train_labels = []

    for subdir in os.listdir(train_path):
        sub_path = os.path.join(train_path, subdir)
        if os.path.isdir(sub_path):
            label = 0
            if subdir == 'Coast':
                label = 1
            elif subdir == 'Forest':
                label = 2
            for sub_folder in os.listdir(sub_path):
                eligible = ['200']
                if sub_folder in eligible:
                    sub_folder_path = os.path.join(sub_path, sub_folder)
                    for file in os.listdir(sub_folder_path):
                        img_path = os.path.join(sub_folder_path, file)
                        #print(f"img_path: {img_path}")
                        img = cv2.imread(img_path)
                        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        sift = cv2.SIFT_create()
                        keypoints, descriptor = sift.detectAndCompute(img, None, 128)
                        #print(descriptor.flatten())
                        #descriptors = np.vstack((descriptors, descriptor.flatten()))
                        descriptors.append(descriptor.flatten())
                        target_lables_sift_train.append(label)
                        if len(keypoints) < 1:
                            continue
                        img_features_sift_train.append(descriptors)
    
    knn = cv2.ml.KNearest_create()
    #train_data = np.array(img_features_sift_train)
    train_labels = np.array(target_lables_sift_train)

    #train_data.reshape(-1, 128)
    #train_labels.reshape(-1, 128)
    print(len(descriptors))
    max_length = len(max(descriptors for d in descriptors))
    padded_desc = [des + [0] * (max_length - len(des)) for des in descriptors]

    knn.train(padded_desc, cv2.ml.ROW_SAMPLE, train_labels)

    test_path = '/Users/sanjith/Desktop/cv-term-project/ProjData/Test'
    test_data = []
    for subdir in os.listdir(test_path):
        sub_path = os.path.join(test_path, subdir)
        if os.path.isdir(sub_path):
            label = 0
            if subdir == 'Coast':
                label = 1
            elif subdir == 'Forest':
                label = 2
            for file in os.listdir(test_path):
                actual_labels.append(label)
                img_path = os.path.join(test_path, file)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                if len(keypoints) < 1:
                    continue
                test_data.append(descriptors)

    test_data = np.array(test_data)
    test_data.reshape(-1, 128)
    retval, results, neigh_resp, dists = knn.findNearest(test_data, k=3)
    print(results)
    pass

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

    # Extract SIFT features from the training images
    # loop through resized/bedroom/50 and resized/bedroom/200 and call sift and store in resized/bedroom/sift
    """
    parent_folder = '/Users/sanjith/Desktop/cv-term-project/ProjData/resized'
    for folder_name in os.listdir(parent_folder):
        for sub_folder in os.listdir(os.path.join(parent_folder, folder_name)):
            sub_folder_list = ['200', '50']
            if sub_folder in sub_folder_list:
                folder_path = os.path.join(parent_folder, folder_name, sub_folder)
            else:
                continue

            if os.path.isdir(folder_path):
            # Loop through the files in the current folder
                for filename in os.listdir(folder_path):
                    # Perform operations on the file
                    img_path = os.path.join(folder_path, filename)

                    extract_sift_features(img_path, filename, folder_name)
    """

    # Train classifiers with different representations and methods
    # ...

    #train_classifier_nn()
    #train_classifier_sift_nn()
    train_and_test_sift_NN()


    # Test the classifiers using test images and report results
    # ...
    
    return


if __name__ == "__main__":
    main()
