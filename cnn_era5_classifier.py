import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# Setting a fixed seed helps us get the same random results each time we run the code.

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# File paths for the original ERA5 images, labels, and GAN-generated images.

DATA_DIR = 'data'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_PATH = os.path.join(DATA_DIR, 'era5_images.npy')
LBL_PATH = os.path.join(DATA_DIR, 'era5_labels.npy')
GAN_PATH = os.path.join(DATA_DIR, 'gan_synthetic_images.npy')

# Basic training settings for the CNN model.

EPOCHS = 30
BATCH_SIZE = 32
PATIENCE = 5


def load_data():

    # Load the original ERA5 image dataset.
    X = np.load(IMG_PATH).astype('float32')

    # Load the class labels for each image.
    y = np.load(LBL_PATH).astype('int32')

    # If GAN images exist, load them too. Otherwise keep it as None.
    X_gan = np.load(GAN_PATH).astype('float32') if os.path.exists(GAN_PATH) else None
    return X, y, X_gan


def standardize_from_train(X_train, X_val, X_test):

    # Calculate mean and standard deviation only from training data.
    # This is important so that validation and test data do not leak into training.
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8

    # Standardize all splits using the training mean and std.
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def build_cnn(input_shape=(17, 21, 5)):
    # L2 regularization is added to reduce overfitting.
    reg = tf.keras.regularizers.l2(1e-4)

    # This is the CNN model used for classifying ERA5 weather maps.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # First convolution block: learns basic spatial patterns.
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Second convolution block: learns more detailed features.
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.30),

        # Third convolution block: learns higher-level feature patterns.
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),

        # Dense layer combines the extracted features before final prediction.
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.30),

        # Final output layer for binary classification: extreme vs normal.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# Compile the model with Adam optimizer, binary crossentropy loss,
# and track both accuracy and AUC during training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


def eval_metrics(y_true, y_prob, threshold=0.5):

    # Convert predicted probabilities into 0 or 1 class labels.
    y_pred = (y_prob >= threshold).astype(int)

    # Get values from the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Standard evaluation metrics.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # Extra weather-event metrics used in this project.
    specificity = tn / (tn + fp) if (tn + fp) else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) else 0
    bias = (tp + fp) / (tp + fn) if (tp + fn) else 0
    far = fp / (tp + fp) if (tp + fp) else 0

# Return all metrics in one dictionary for easy saving and comparison.
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'csi': csi,
        'bias': bias,
        'false_alarm_ratio': far,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def train_once(X_train, y_train, X_val, y_val):

    # Build a fresh CNN model.
    model = build_cnn(input_shape=X_train.shape[1:])

# Compute class weights because extreme events are usually fewer than normal events.
# This helps the model pay more attention to the minority class
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}

# EarlyStopping stops training when validation AUC stops improving.
# ReduceLROnPlateau lowers the learning rate when validation loss gets stuck.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

# Train the model using training data and validate on validation data.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        class_weight=class_weight,
        callbacks=callbacks
    )
    return model, history


def main():
    # Load original data and optional GAN-augmented data.
    X, y, X_gan = load_data()

# Print basic dataset information.
    print("ERA5 images shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Extreme proportion:", y.mean())

# First split: keep 20% of the data as final test set.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
# Second split: from the remaining data, create training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=SEED, stratify=y_trainval
    )
# Standardize all splits using only training statistics.
    X_train_s, X_val_s, X_test_s, mean, std = standardize_from_train(X_train, X_val, X_test)

# -----------------------------
# Baseline CNN without GAN data
# -----------------------------
    print("\nTraining baseline CNN...")
    baseline_model, hist_base = train_once(X_train_s, y_train, X_val_s, y_val)

    # Predict probabilities on the test set.
    prob_base = baseline_model.predict(X_test_s, verbose=0).ravel()

    # Calculate test metrics for the baseline model.
    met_base = eval_metrics(y_test, prob_base)

# Save row-level baseline predictions.
    pd.DataFrame({
        'y_true': y_test,
        'y_prob': prob_base,
        'y_pred': (prob_base >= 0.5).astype(int)
    }).to_csv(os.path.join(RESULTS_DIR, 'cnn_results_baseline.csv'), index=False)

# Save training history values for later plotting.
    histories = {
        'baseline_loss': hist_base.history['loss'],
        'baseline_val_loss': hist_base.history['val_loss'],
        'baseline_auc': hist_base.history['auc'],
        'baseline_val_auc': hist_base.history['val_auc']
    }

# Store the baseline metrics in a list.
    metrics_rows = [dict(model='baseline', **met_base)]


# --------------------------------
# CNN with GAN-augmented train data
# --------------------------------
    if X_gan is not None:
        print("\nGAN synthetic images found:", X_gan.shape)

        # Standardize GAN images using the same training mean and std.
        X_gan_s = (X_gan - mean) / std

        # GAN images are treated as extreme-event samples.
        y_gan = np.ones(len(X_gan_s), dtype='int32')

        # Add GAN samples to the original training set.
        X_train_aug = np.concatenate([X_train_s, X_gan_s], axis=0)
        y_train_aug = np.concatenate([y_train, y_gan], axis=0)

        print("\nTraining augmented CNN...")
        aug_model, hist_aug = train_once(X_train_aug, y_train_aug, X_val_s, y_val)

        # Predict on the same test set for fair comparison.
        prob_aug = aug_model.predict(X_test_s, verbose=0).ravel()
        met_aug = eval_metrics(y_test, prob_aug)

        # Save row-level predictions for the augmented model.
        pd.DataFrame({
            'y_true': y_test,
            'y_prob': prob_aug,
            'y_pred': (prob_aug >= 0.5).astype(int)
        }).to_csv(os.path.join(RESULTS_DIR, 'cnn_results_augmented.csv'), index=False)

        # Add augmented metrics to the comparison list.
        metrics_rows.append(dict(model='augmented', **met_aug))

        # Add augmented training history too.
        histories.update({
            'aug_loss': hist_aug.history['loss'],
            'aug_val_loss': hist_aug.history['val_loss'],
            'aug_auc': hist_aug.history['auc'],
            'aug_val_auc': hist_aug.history['val_auc']
        })

        # Save the final augmented model.
        aug_model.save(os.path.join(RESULTS_DIR, 'cnn_era5_classifier.keras'))
    else:
        
        # If no GAN file is found, only baseline training is done.
        print("\nNo gan_synthetic_images.npy found. Only baseline run will be done.")
        baseline_model.save(os.path.join(RESULTS_DIR, 'cnn_era5_classifier.keras'))

    # Save all final metrics in one CSV file.
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'cnn_metrics_summary.csv'), index=False)

    # Save training history as a JSON file.
    with open(os.path.join(RESULTS_DIR, 'cnn_histories.json'), 'w') as f:
        json.dump(histories, f)

# ---------------------------------
# Plot validation loss and val AUC
# ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist_base.history['val_loss'], label='Baseline val loss', color='steelblue')
    if 'aug_val_loss' in histories:
        axes[0].plot(histories['aug_val_loss'], label='Augmented val loss', color='coral')
    axes[0].set_title('Validation loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist_base.history['val_auc'], label='Baseline val AUC', color='steelblue')
    if 'aug_val_auc' in histories:
        axes[1].plot(histories['aug_val_auc'], label='Augmented val AUC', color='coral')
    axes[1].set_title('Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cnn_training_comparison.png'), dpi=180)
    plt.close()

# ---------------------------------
# Plot confusion matrices for comparison
# ---------------------------------
    fig, axes = plt.subplots(1, len(metrics_rows), figsize=(5 * len(metrics_rows), 4))
    if len(metrics_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, metrics_rows):
        cm = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]])
        ax.imshow(cm, cmap='Blues')
        ax.set_title(f"{row['model'].title()} confusion")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Extreme'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Normal', 'Extreme'])

 # Write the confusion matrix values inside each cell.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cnn_confusion_comparison.png'), dpi=180)
    plt.close()

# Print the final summary table in the console.
    print("\nFinal metrics:")
    print(metrics_df)


# Run the full pipeline when this file is executed directly.
if __name__ == '__main__':
    main()
