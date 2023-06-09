import tensorflow as tf
from keras.applications import MobileNetV3Small
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
import os

# Load the MobileNetV3_small model without the top classification layer:
base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the base model layers:
base_model.trainable = False

# Create a new model with the base model and a new classification layer for 3 classes:
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
 
# Compile the model with a suitable optimizer and loss function:
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare your data using data augmentation with rotations:
train_datagen = ImageDataGenerator(
    rotation_range=20,
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    'path/to/train/directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Set up learning rate decay, early stopping, and TensorBoard callbacks:
def lr_decay(epoch):
    initial_lr = 0.001
    decay_rate = 0.1
    decay_step = 10
    return initial_lr * (decay_rate ** (epoch // decay_step))

lr_scheduler = LearningRateScheduler(lr_decay)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
log_dir = os.path.join("logs", "fit", "mobilenetv3_small")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [lr_scheduler, early_stopping, tensorboard_callback]

# Train the model with the augmented data and callbacks:
model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=callbacks
)
