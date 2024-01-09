import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

dataset = [
    {"class": "cat", "image": cv2.imread("image/photo_1.jpg")},
    {"class": "cat", "image": cv2.imread("image/photo_2.jpg")},
    {"class": "cat", "image": cv2.imread("image/photo_3.jpg")},
    {"class": "cat", "image": cv2.imread("image/photo_4.jpg")},
    {"class": "cat", "image": cv2.imread("image/photo_5.jpg")},
    {"class": "dog", "image": cv2.imread("image/photo_6.jpg")},
    {"class": "dog", "image": cv2.imread("image/photo_7.jpg")},
    {"class": "dog", "image": cv2.imread("image/photo_8.jpg")},
    {"class": "dog", "image": cv2.imread("image/photo_9.jpg")},
    {"class": "dog", "image": cv2.imread("image/photo_10.jpg")},
    {"class": "car", "image": cv2.imread("image/photo_11.jpg")},
    {"class": "car", "image": cv2.imread("image/photo_12.jpg")},
    {"class": "car", "image": cv2.imread("image/photo_13.jpg")},
    {"class": "bus", "image": cv2.imread("image/photo_14.jpg")},
]

common_size = (224, 224)

image_vectors = []
class_labels = []
for item in dataset:
    image = cv2.cvtColor(item["image"], cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, common_size)
    vector = resized_image.flatten().reshape(1, -1)
    image_vectors.append(vector)
    class_labels.append(item["class"])

image_vectors = np.concatenate(image_vectors, axis=0)
class_labels = np.array(class_labels)

# Building the Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn_model.fit(image_vectors)

# Test dataset
test_dataset = [
    {"class": "cat", "image": cv2.imread("image/test_1.jpg")},
    {"class": "dog", "image": cv2.imread("image/test_2.jpg")},
    {"class": "car", "image": cv2.imread("image/test_3.jpg")},
]

for test_item in test_dataset:
    test_image = cv2.cvtColor(test_item["image"], cv2.COLOR_BGR2GRAY)
    resized_test_image = cv2.resize(test_image, common_size)
    test_vector = resized_test_image.flatten().reshape(1, -1)

    distances, indices = knn_model.kneighbors(test_vector)

    print(f"Test Image: {test_item['class']}")
    print("Nearest Images:")
    for i, index in enumerate(indices.flatten()):
        print(
            f"  - Class: {class_labels[index]}, Distance: {distances.flatten()[i]}, Index: {index}"
        )
