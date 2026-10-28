import tensorflow as tf

from preprocess import train_gen, X_val, y_val, X_train_n, X_train_p, batch_size
from model import model

######################################
# Train
######################################
steps_per_epoch = (len(X_train_n) + len(X_train_p)) // batch_size
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("resnet_model.h5", save_best_only=True),
]


trained_model = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

