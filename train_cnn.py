import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. Load dataset from folders ---

data_dir = "tennis_data"   # folder with tennis_ball/ and not_tennis_ball/

img_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,      # 80% train, 20% val
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Optional: performance tweaks
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# --- 2. Build a small CNN from scratch ---

model = keras.Sequential([
    layers.Rescaling(255./255, input_shape=img_size + (3,)),  # normalize [0,255] -> [0,1]

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # 1 output neuron for binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 3. Train the model ---

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 4. Save the trained model ---

model.save("tennis_cnn.h5")
print("Model saved as tennis_cnn.h5")
