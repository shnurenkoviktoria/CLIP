# Image Classification with Nearest Neighbors and CLIP

This README demonstrates image classification using both the Nearest Neighbors algorithm and the CLIP (Contrastive Language-Image Pre-Training) model.

## Image Classification with Nearest Neighbors

### Dataset

- The dataset consists of images belonging to different classes such as cat, dog, car, and bus.
- Images are loaded from the `image` directory and stored in a list of dictionaries containing image arrays and class labels.

### Preprocessing

- Images are resized to a common size of 224x224 pixels.
- Grayscale conversion is applied to reduce dimensionality.
- Image vectors are flattened and concatenated to form the feature matrix.
- Class labels are stored separately.

### Nearest Neighbors Model

- A Nearest Neighbors model is built using the Euclidean distance metric and 10 neighbors.
- The model is trained on the feature matrix consisting of image vectors.

### Testing

- Test images are loaded from the `image` directory.
- Each test image is resized, converted to grayscale, flattened, and used to query the Nearest Neighbors model.
- Nearest images and their corresponding classes, distances, and indices are printed for each test image.

### Dependencies

- OpenCV (cv2)
- NumPy
- scikit-learn

## Image Classification with CLIP

### CLIP Model

- The CLIP model is loaded with the ViT-B/32 architecture.
- A test image is tokenized and passed along with prompts to the CLIP model.
- The model predicts probabilities for different labels based on the image and prompts.

### Testing

- A test image is loaded from the `image` directory.
- A set of prompts related to cats is defined.
- Each prompt is tokenized and passed to the CLIP model along with the test image.
- The model outputs label probabilities and the predicted label based on the prompts.

### Dependencies

- OpenAI CLIP
- NumPy
- PyTorch
- PIL (Python Imaging Library)
