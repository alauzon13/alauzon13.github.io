from tensorflow.keras.applications import EfficientNetB0
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import shutil
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator 




def split_data(df_path):
    """Function that splits image data into training, testing, and validation."""
    # Load data 
    data = pd.read_csv(df_path)

    X_paths = data["Path"]
    y = data["Label"]

    # Split into training and testing 
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(X_paths, y, test_size=0.2, random_state=1)

    X_train_paths, X_val_paths, y_train, y_val  = train_test_split(X_train_paths, y_train, test_size=0.25, random_state=1)

    # Create new folders for train and test data 
    os.makedirs("Modelling/Data/Train/Real", exist_ok=True)
    os.makedirs("Modelling/Data/Train/Fake", exist_ok=True)

    os.makedirs("Modelling/Data/Test/Real", exist_ok=True)
    os.makedirs("Modelling/Data/Test/Fake", exist_ok=True)

    os.makedirs("Modelling/Data/Val/Real", exist_ok=True)
    os.makedirs("Modelling/Data/Val/Fake", exist_ok=True)

    for file_path in X_train_paths:
        if "Real" in file_path:
            shutil.move(file_path, "Modelling/Data/Train/Real")
        else:
            shutil.move(file_path, "Modelling/Data/Train/Fake")

    for file_path in X_test_paths:
        if "Real" in file_path:
            shutil.move(file_path, "Modelling/Data/Test/Real")
        else:
            shutil.move(file_path, "Modelling/Data/Test/Fake")



    for file_path in X_val_paths:
        if "Real" in file_path:
            shutil.move(file_path, "Modelling/Data/Val/Real")
        else:
            shutil.move(file_path, "Modelling/Data/Val/Fake")  



def data_augmentation(train_dir, val_dir, test_dir):
    # Augmentation

    train_datagen = ImageDataGenerator(
        rescale=1. / 255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(224, 224)
    )

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(224, 224)
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir, 
        batch_size = 20,
        class_mode = 'binary',
        target_size = (224, 224)
    )

    return(train_generator, validation_generator, test_generator)


def model_developer(train_generator, validation_generator):
    # Load Base Model 

    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )


    # Freeze Layers

    for layer in base_model.layers:
        layer.trainable = False

    # Build Model

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    predictions = Dense(1, activation="sigmoid")(x)
    model_final = Model(inputs=base_model.input, outputs=predictions)


    model_final.compile(
        optimizer = RMSprop(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    eff_history = model_final.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100,
        epochs=20
    )

    return(model_final, eff_history)
    







