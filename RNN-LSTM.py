# Section 1: Import Libraries and Load Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Permute, Multiply, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\rithi\\OneDrive\\Desktop\\backend\\creditcard_2023.csv'
data = pd.read_csv(file_path)

# Check for missing values
if data.isnull().sum().sum() > 0:
    data = data.dropna()
# Section 2: Filter Transactions and Prepare Data
# Filter the transactions where is_online is True
online_data = data[data['is_online'] == True]

# Features and target variable
X = online_data.drop(columns=['Class', 'is_online'])
y = online_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
# Section 3: Define Attention Mechanism
# Attention mechanism
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(input_dim, activation='softmax')(a)
    a = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a])
    return Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
# Section 4: Build and Compile the Model
# Build the model
inputs = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
lstm_out = LSTM(units=50, return_sequences=True)(inputs)
attention_out = attention_3d_block(lstm_out)
output = Dense(1, activation='sigmoid')(attention_out)

model = Model(inputs=[inputs], outputs=[output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
# Section 5: Train the Model
# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
# Section 6: Evaluate the Model
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred = model.predict(X_test_reshaped)
y_pred_class = (y_pred > 0.5).astype(int)

# Visualize the results
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.yticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.show()

# Classification Report
print(classification_report(y_test, y_pred_class, target_names=['Non-Fraud', 'Fraud']))
# Section 7: Display Results
# Display the results
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_class.flatten()})
fraud_transactions = results[results['Predicted'] == 1]
non_fraud_transactions = results[results['Predicted'] == 0]

print("Fraud Transactions:\n", fraud_transactions)
print("Non-Fraud Transactions:\n", non_fraud_transactions)

# Visualize the balance of the dataset
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()from sklearn.decomposition import PCA

# Combine the training and test datasets for visualization
X_combined = np.concatenate((X_train_scaled, X_test_scaled), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)

# Sample the data for faster computation
sample_size = 1000  # Adjust this size based on your machine's capabilities
np.random.seed(42)
sample_indices = np.random.choice(X_combined.shape[0], sample_size, replace=False)
X_sample = X_combined[sample_indices]
y_sample = y_combined[sample_indices]

# Perform PCA to reduce dimensionality before t-SNE
pca = PCA(n_components=30)  # Reduce to 30 components before applying t-SNE
X_pca = pca.fit_transform(X_sample)

# Perform t-SNE
tsne = TSNE(n_components=3, random_state=42, perplexity=30)  # Increase perplexity for faster convergence
X_tsne = tsne.fit_transform(X_pca)

# Plot the results
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot non-fraud transactions
ax.scatter(X_tsne[y_sample == 0, 0], X_tsne[y_sample == 0, 1], X_tsne[y_sample == 0, 2], 
           c='blue', label='Non-Fraud', alpha=0.5)

# Plot fraud transactions
ax.scatter(X_tsne[y_sample == 1, 0], X_tsne[y_sample == 1, 1], X_tsne[y_sample == 1, 2], 
           c='red', label='Fraud', alpha=0.5)

ax.set_title('3D t-SNE Visualization of Fraudulent Transactions')
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.set_zlabel('t-SNE Feature 3')
ax.legend()
plt.show()
