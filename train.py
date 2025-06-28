import os
from PIL import Image
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Paths to your image folders
cat_folder = 'D:\internship\SVM\Cat (1)'
dog_folder = 'D:\internship\SVM\Dog'

X = []
y = []

# Load cat images
for filename in os.listdir(cat_folder):
    img = Image.open(os.path.join(cat_folder, filename)).convert('L')
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0

    features, _ = hog(
        img_array,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    X.append(features)
    y.append(1)  # Label for cat

# Load dog images
for filename in os.listdir(dog_folder):
    img = Image.open(os.path.join(dog_folder, filename)).convert('L')
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0

    features, _ = hog(
        img_array,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    X.append(features)
    y.append(0)  # Label for dog

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, 'model.sav')
print("Model saved as model.sav")
