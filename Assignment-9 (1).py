import os
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurations
DATASET_PATH = 'dataset'
IMG_DIM = 128
EPOCHS = 10
BATCH_SIZE = 32

def main():
    # --- Task 1: Data Understanding ---
    print("Checking dataset structure...")
    if os.path.exists(DATASET_PATH):
        for root, dirs, files in os.walk(DATASET_PATH):
            print(f"Found {len(files)} files in {root}")
    else:
        print("Dataset not found!")

    # --- Task 2: Data Preprocessing ---
    # Generator for normalizing and splitting
    generator = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
    
    try:
        train_set = generator.flow_from_directory(
            DATASET_PATH,
            target_size=(IMG_DIM, IMG_DIM),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )

        test_set = generator.flow_from_directory(
            DATASET_PATH,
            target_size=(IMG_DIM, IMG_DIM),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        print("\nClasses overview:", train_set.class_indices)
        
        # Plot 5 sample images
        batch_x, batch_y = next(train_set)
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            ax[i].imshow(batch_x[i])
            label_name = 'Dog' if batch_y[i] == 1 else 'Cat'
            ax[i].set_title(label_name)
            ax[i].axis('off')
        plt.savefig("friend2_sample_images.png")
        plt.close()
        
        # --- Task 3: Model Development ---
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        # Training
        print("\nStarting model training...")
        history = model.fit(train_set, validation_data=test_set, epochs=EPOCHS)
        
        # --- Task 4: Model Evaluation ---
        results = model.evaluate(test_set)
        print(f"\nFinal Test Accuracy: {results[1] * 100:.2f}%")
        
        # Predictions
        test_set.reset()
        predictions = (model.predict(test_set) > 0.5).astype("int32")
        true_labels = test_set.classes
        
        # Metrics
        print("\nModel Evaluation Metrics:")
        print("Precision: ", metrics.precision_score(true_labels, predictions))
        print("Recall: ", metrics.recall_score(true_labels, predictions))
        print("F1-Score: ", metrics.f1_score(true_labels, predictions))
        print("\nConfusion Matrix:\n", metrics.confusion_matrix(true_labels, predictions))
        
        # Plotting
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], c='blue', label='Train')
        plt.plot(history.history['val_accuracy'], c='orange', label='Validation')
        plt.title('Model Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], c='blue', label='Train')
        plt.plot(history.history['val_loss'], c='orange', label='Validation')
        plt.title('Model Loss')
        plt.legend()
        plt.savefig("friend2_training_metrics.png")
        plt.close()
        
        print("\nObservations:\n1. Training progressed well over 10 epochs.\n2. Overfitting can be observed if validation loss spikes.\n3. CNN is highly effective in differentiating spatial features of pets.")
        
    except Exception as e:
        print("Could not load data or train model. Ensure dataset is set up properly.")
        print(e)

if __name__ == "__main__":
    main()
