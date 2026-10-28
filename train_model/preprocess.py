import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input

######################################
# 1. Load & preprocess images
######################################
def load_and_preprocess(image_paths, label, img_size=(224, 224)):
    X, y = [], []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype("float32"))  # use ResNet preprocessor
        X.append(img)
        y.append(label)
    return np.array(X), np.array(y)

######################################
# 2. Load & split training data
######################################
norm_paths = glob.glob("./chest_xray/train/NORMAL/*.jpeg")
pneu_paths = glob.glob("./chest_xray/train/PNEUMONIA/*.jpeg")

X_norm, y_norm = load_and_preprocess(norm_paths, label=0)
X_pneu, y_pneu = load_and_preprocess(pneu_paths, label=1)

# Undersample PNEUMONIA to 2500
X_pneu_under = X_pneu[:2500]
y_pneu_under = y_pneu[:2500]

# Train/validation split
X_train_n, X_val_n, y_train_n, y_val_n = train_test_split(
    X_norm, y_norm, test_size=0.2, random_state=42
)

X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
    X_pneu_under, y_pneu_under, test_size=0.2, random_state=42
)

# Combine validation sets
X_val = np.concatenate([X_val_n, X_val_p], axis=0)
y_val = np.concatenate([y_val_n, y_val_p], axis=0)

######################################
# 3. Generators (balanced)
######################################
batch_size = 32
half_bs = batch_size // 2

aug = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

gen_norm = aug.flow(X_train_n, y_train_n, batch_size=half_bs, shuffle=True)
gen_pneu = aug.flow(X_train_p, y_train_p, batch_size=half_bs, shuffle=True)


def balanced_generator(gen0, gen1):
    while True:
        X0, y0 = next(gen0)
        X1, y1 = next(gen1)
        Xb = np.vstack([X0, X1])
        yb = np.concatenate([y0, y1])
        idx = np.random.permutation(len(yb))
        yield Xb[idx], yb[idx]




train_gen = balanced_generator(gen_norm, gen_pneu)