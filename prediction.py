import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

# Load dataset
(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Normalize and reshape test images
x_test_norm = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Make predictions for the first 2 images
predictions = model.predict(x_test_norm[:2])
predicted_classes = np.argmax(predictions, axis=1)

# Class labels for Fashion MNIST
class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Plot the results
for i in range(2):
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Predicted: {class_labels[predicted_classes[i]]} | Actual: {class_labels[y_test[i]]}")
    plt.axis("off")
    plt.show()
