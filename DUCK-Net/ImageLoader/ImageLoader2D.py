import glob

import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

#folder_path = "/kaggle/input/dataset-ducknet/Kvasir-SEG-20241223T142106Z-001/Kvasir-SEG/"  # Add the path to your data directory


def load_data(img_height, img_width, folder_path):
    IMAGES_PATH = folder_path + 'images/'
    MASKS_PATH = folder_path + 'masks/'
    print('IMAGES_PATH: ',IMAGES_PATH)
    print('MASKS_PATH: ',MASKS_PATH)
    
    train_ids = glob.glob(IMAGES_PATH + "*.jpg")
    # if dataset == 'kvasir':
    #     train_ids = glob.glob(IMAGES_PATH + "*.jpg")
        
    # if dataset == 'cvc-clinicdb':
    #     train_ids = glob.glob(IMAGES_PATH + "*.tif")

    # if dataset == 'cvc-colondb' or dataset == 'etis-laribpolypdb':
    #     train_ids = glob.glob(IMAGES_PATH + "*.png")

    # if images_to_be_loaded == -1:
    images_to_be_loaded = len(train_ids)
    
    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print('Resizing training images and masks: ' + str(images_to_be_loaded))
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")
        
        image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
        mask_ = Image.open(mask_path).convert("L")  # Convert mask to grayscale
        
        # Resize images using PIL
        image = image.resize((img_height, img_width), resample=Image.LANCZOS)
        mask_ = mask_.resize((img_height, img_width), resample=Image.LANCZOS)
        
        # Convert images to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize image
        mask_ = np.array(mask_, dtype=np.uint8)  # Keep mask as integers
        
        # Create an empty boolean mask
        mask = np.zeros((img_height, img_width), dtype=bool)
        
        # Store the processed image in X_train
        X_train[n] = image

        for i in range(img_height):
            for j in range(img_width):
                if mask_[i, j] >= 127:
                    mask[i, j] = 1

        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1)

    return X_train, Y_train
