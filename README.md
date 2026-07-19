# MNIST Handwritten Digit Classifier and Drawer

This project implements two deep learning architectures (Multi-Layer Perceptron and Convolutional Neural Network) to classify handwritten digits (0 to 9) using PyTorch. Additionally, it features an interactive Tkinter graphical application that allows users to draw digits and see the predictions in real time.

## Project Structure

* mnistMLP.ipynb: Notebook detailing the implementation and training of a Multi-Layer Perceptron.
* mnistCNN.ipynb: Notebook detailing the implementation and training of a Convolutional Neural Network.
* handdrawnDigitClassification.py: An interactive desktop application built with Tkinter that loads the trained CNN weights for real-time digit prediction.
* MNIST_MLPmodel_weights.pth: Pre-trained weights for the MLP model.
* MNIST_CNNmodel_weights.pth: Pre-trained weights for the CNN model.

## Model Architectures

This project compares two neural network approaches:

### 1. Multi-Layer Perceptron (MLP)
The MLP serves as the baseline model. It flattens the 28x28 input images into a 784-dimensional vector and passes them through fully connected layers:
* Flatten
* Linear(784 to 128) -> ReLU
* Linear(128 to 64) -> ReLU
* Linear(64 to 10)

This architecture is lightweight but lacks spatial awareness, meaning it treats pixel adjacencies the same way regardless of structure.

### 2. Convolutional Neural Network (CNN)
The CNN captures local spatial patterns (edges, loops, curves) using convolutional operations:
* Convolutional Block 1: Conv2d(1, 32, kernel_size=3) -> BatchNorm -> ReLU -> MaxPool2d(2)
* Convolutional Block 2: Conv2d(32, 64, kernel_size=3) -> BatchNorm -> ReLU -> MaxPool2d(2) -> Dropout2d(0.25)
* Classifier: Flatten -> Linear(64 * 7 * 7, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 10)

The CNN achieves 99.03 percent test accuracy on the MNIST dataset.

## Interactive Drawing Application

The interactive GUI is built using Python's standard Tkinter library and Pillow:
* Draw: Users draw on a black 400x400 canvas.
* Process: The canvas image is resized down to 28x28 pixels using Lanczos interpolation and normalized using MNIST dataset statistics (mean of 0.1307, standard deviation of 0.3081).
* Predict: The processed tensor is fed to the pre-trained CNN model. The application outputs the predicted digit along with the softmax confidence percentage.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/VecetyP/MNIST-Classifier.git
   cd path/to/your/MNIST/Classifier/folder
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On macOS/Linux use: source .venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: On some Linux distributions, you may need to install Tkinter system-wide:
   ```bash
   sudo apt-get install python3-tk
   ```

4. Run the drawing application:
   ```bash
   python handdrawnDigitClassification.py
   ```

5. Explore the models:
   Open the notebooks `mnistMLP.ipynb` or `mnistCNN.ipynb` using Jupyter Notebook or your preferred IDE to view model training details.
