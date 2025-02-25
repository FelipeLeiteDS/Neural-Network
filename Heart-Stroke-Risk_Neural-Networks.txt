# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
import csv

# Load the dataset
df = pd.read_excel("/heart.xlsx")

# Split features
X = df.drop('output', axis=1)
y = df['output']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=input_shape),
        keras.layers.Dense(515, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model

# Create and compile the model
model = create_model(input_shape=[X_train.shape[1]])
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=15,
    epochs=50,
    callbacks=[early_stopping],
    verbose=0,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print classification report
print(classification_report(y_test, y_pred))

# Example prediction for a single input
example_input = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
prediction = model.predict(example_input)
print(f"Prediction for example input: {prediction[0][0]:.4f}")

# Analyze the effect of a specific variable on predictions
def analyze_variable_effect(model, X_train, variable_name, output_file):
    mean_values = X_train.mean()
    predictions = []

    for value in range(0, 7):
        input_values = mean_values.copy()
        input_values[variable_name] = value
        prediction = model.predict(np.array([input_values]))
        predictions.append((variable_name, value, prediction[0][0]))

    # Save predictions to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Variable', 'Value', 'Prediction'])  # Write header
        writer.writerows(predictions)

# Analyze the effect of 'oldpeak' on predictions
analyze_variable_effect(model, X_train, 'oldpeak', 'predictions.csv')
