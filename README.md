

# Fashion MNIST Image Classification with CNN

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** using the **Fashion MNIST dataset** with both **Python (Keras)** and **R (Keras interface)**. It serves as a prototype for a larger project on **profile image-based product targeting**.

---

## 📁 Project Structure

```
fashion-mnist-cnn/
│
├── cnnmodel.py         # Main Python script (model training and prediction)
├── model.R          # R version using keras package
├── prediction.py          # Python script to make predictions
├── README.md              # Project documentation
```

---

## 📌 Objectives

* Train a CNN to classify images of clothing items from Fashion MNIST.
* Use 6-layer architecture with convolutional and pooling layers.
* Make predictions on unseen data.
* Prepare the system for future adaptation to user profile image classification.

---

## 🧪 Dataset

The **Fashion MNIST** dataset is a collection of 70,000 grayscale 28x28 pixel images in 10 fashion categories:

* T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## 🏗️ Model Architecture

A 6-layer CNN:

1. Conv2D (32 filters)
2. MaxPooling2D
3. Conv2D (64 filters)
4. MaxPooling2D
5. Flatten
6. Dense (128 neurons) + Output (Softmax)

---

## 🚀 Steps on How to Run

1. Clone the repository or copy the script.
2. Ensure Python and required libraries are installed:

   ```bash
   pip install tensorflow matplotlib numpy
3. Run the script:

   ```bash
   python cnnmodel.py
   ```
4. Install all required packages for R:

   ```bash
   install.packages("keras")
   keras::install_keras()
5. Run the script:

   ```bash
   Rscript mmodel.R
6. Run the script:

   ```bash
   python prediction.py
   ```

## 🧾 Output

* Trained CNN model
* Accuracy on test dataset
* Predictions on sample images with labels and visualization

---

## Future Scope

* Replace Fashion MNIST with real user profile images.
* Train on custom dataset for targeted marketing.

---
## 📄 License

This project is open source and free to use for educational or internal business purposes.

---

## 💬 Contact

For improvements, feel free to open a pull request on my GitHub account or contact me directly - josephdokhare@gmail.com
