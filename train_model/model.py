from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import AUC

######################################
# Build ResNet50 model
######################################
input_shape = (224, 224, 3)
base_model = ResNet50(
    include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
)
base_model.trainable = False  # freeze for faster training

model = models.Sequential(
    [
        base_model,
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")],
)