library(keras)

# Load dataset
fashion_mnist <- dataset_fashion_mnist()
c(x_train, y_train) %<-% fashion_mnist$train
c(x_test, y_test) %<-% fashion_mnist$test

# Preprocess data
x_train <- x_train / 255
x_test <- x_test / 255

dim(x_train) <- c(nrow(x_train), 28, 28, 1)
dim(x_test) <- c(nrow(x_test), 28, 28, 1)

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Build model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28, 28, 1)) %>% # Layer 1
  layer_max_pooling_2d(pool_size = c(2,2)) %>%                                                           # Layer 2
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%                             # Layer 3
  layer_max_pooling_2d(pool_size = c(2,2)) %>%                                                           # Layer 4
  layer_flatten() %>%                                                                                    # Layer 5
  layer_dense(units = 128, activation = 'relu') %>%                                                      # Layer 6
  layer_dense(units = 10, activation = 'softmax')                                                        # Output layer

# Compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Train
model %>% fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split = 0.1)

# Evaluate
score <- model %>% evaluate(x_test, y_test)
cat("Test accuracy:", score$accuracy, "\n")
