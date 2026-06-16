import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


data_dir = r"F:\Sign language detection\Data" # Change this path to your dataset directory (should contain subfolders for each class)


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)


train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)
)
base_model.trainable = False  # freeze pretrained layers

# Add custom layers on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    verbose=1
)

#  Save trained model
model.save("sign_language_model.h5")
print("\nModel saved as 'sign_language_model.h5'")

# Display class labels
print("\nClasses found:")
for label, idx in train_data.class_indices.items():
    print(f"{idx}: {label}")
