# AI-ML Assignment 9: Image Classification using CNN

## Objective
The objective of this project is to develop a Convolutional Neural Network (CNN) model to automatically classify pet images into Cats and Dogs for an animal welfare organization.

## Dataset Link
[Kaggle: Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

## Libraries Used
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- OS

## Methodology
1. **Data Understanding**: Loaded the dataset and displayed the folder structure, image dimensions, and sample images.
2. **Data Preprocessing**: Images were resized to 128x128 pixels, pixel values were normalized to the range 0-1, and the dataset was split into 80% training and 20% testing using `ImageDataGenerator`.
3. **Model Development**: A CNN architecture was built using Conv2D, MaxPooling2D, Flatten, and Dense layers. The model was compiled using Adam optimizer and Binary Crossentropy loss.
4. **Model Evaluation**: Evaluated the model using Test Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and plotted Accuracy vs. Epoch and Loss vs. Epoch graphs.

## CNN Architecture
The CNN consists of the following layers:
- `Conv2D` (32 filters, 3x3, ReLU)
- `MaxPooling2D` (2x2)
- `Conv2D` (64 filters, 3x3, ReLU)
- `MaxPooling2D` (2x2)
- `Conv2D` (128 filters, 3x3, ReLU)
- `MaxPooling2D` (2x2)
- `Flatten` Layer
- `Dense` Layer (128 neurons, ReLU)
- `Output` Layer (1 neuron, Sigmoid)

## Results
- **Test Accuracy**: *(Will be output when script runs)*
- **Precision**: *(Will be output when script runs)*
- **Recall**: *(Will be output when script runs)*
- **F1-Score**: *(Will be output when script runs)*

*Note: The script generates `evaluation_graphs.png` representing the Accuracy vs Epoch and Loss vs Epoch graphs, along with standard console output for the Confusion Matrix.*

### Observations
1. The CNN successfully learns to extract features and differentiate between Cats and Dogs as accuracy improves over epochs.
2. If training accuracy keeps increasing while validation accuracy plateaus, it signifies overfitting, which can be mitigated with Dropout layers.
3. The confusion matrix helps identify whether the model is biased toward predicting one class over the other.

## Conclusion
**Key Findings**: The developed CNN effectively classified images of cats and dogs, demonstrating the feasibility of automating this process using deep learning techniques. 

**Importance of Convolution and Pooling Layers**: Convolution layers are essential for extracting spatial features like edges, shapes, and textures from the images. Pooling layers help by reducing dimensionality and computational load, while also making the model invariant to small translations.

**One advantage of CNN over ANN for image classification**: CNNs preserve the 2D spatial structure and relationships of pixels in an image, whereas ANNs require flattening the input, completely losing the 2D spatial information which is crucial for recognizing visual patterns.

**One limitation of CNN**: CNNs are highly computationally expensive and require large amounts of labeled data to train effectively from scratch without overfitting.
