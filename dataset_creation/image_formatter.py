from PIL import Image, ImageOps
import numpy as np
import os
import shutil
import cv2
import random

def get_ROI(img):
    '''Takes a Pillow (PIL) Image object and gets the region of intrest (ROI) / bounding box
    Resizes Image (Max 24px height or width) -> 2px padding will be added in different method
    Returns resized ROI as PIL Image object'''

    imgArray = np.array(img)
    height, width = imgArray.shape

    # * Get bounding box
    yMin = None
    yMax = None
    xMin = None
    xMax = None

    for y in range(height):
        for x in range(width):

            if (imgArray[y][x] != 0 and yMin is None):
                yMin = y
            if (imgArray[height-y-1][x] != 0 and yMax is None):
                yMax = height-y

    for x in range(width):
        for y in range(height):

            if (imgArray[y][x] != 0 and xMin is None):
                xMin = x
            if (imgArray[y][width-x-1] != 0 and xMax is None):
                xMax = width-x

    roiArray = imgArray[yMin:yMax, xMin:xMax]
    roi_img = Image.fromarray(roiArray)

    return roi_img

def resize_image(img, max):
    '''Resizes image to determined (max) size.
    The resized image will either have a height or width of the given max.
    The aspect ratio of the image image is preserved'''

    width, height = img.size

    if (height > width):
        ratio = max/height

        newWidth = int(ratio*width+0.5)
        if newWidth == 0:
            newWidth = 1

        resized_img = img.resize( (newWidth, max), resample = 1)
    else:
        ratio = max/width

        newHeight = int(ratio*height+0.5)
        if newHeight == 0:
            newHeight = 1

        resized_img = img.resize( (max, newHeight),  resample = 1)

    return resized_img

def pad_image(img, max):
    '''Pad image to be square with given max heights and widths'''

    width, height = img.size

    wDelta = max - width
    hDelta = max - height

    padded_img = ImageOps.expand(img, border = (int(wDelta/2), int(hDelta/2), int(wDelta/2+0.5), int(hDelta/2+0.5)), fill=0)

    return padded_img

def format_image(fp, targetDir):
    '''Get ROI, resize image, and pad to apropriate format.'''

    img = Image.open(fp)
    img = get_ROI(img)
    img = resize_image(img, 24)
    img = pad_image(img, 28)

    fileName = fp.split("/")[-1]

    img.save(targetDir + "/" + fileName)

def format_all(imageDir, targetDir):
    '''Takes all images in "imageDir" and formats them. Puts these into "targetDir"'''

    imgFiles = os.listdir(imageDir)
    finished = os.listdir(targetDir)

    for f in imgFiles:
        if f not in finished:
            print(str(f))
            format_image(imageDir + "/" + f, targetDir)

def generate_number_dataset(imgDir, targetDir):
    '''Generates the dataset for number glyphs.
    Also randomizes order of glyphs'''
    
    imgFiles = os.listdir(imgDir)
    sortedFiles = []

    # * Sort out glyph types to only incude numbers
    for f in imgFiles:
        if f.split('_')[1] == 'num':
            sortedFiles.append(f)

    # * Randomize glyphs
    random.shuffle(sortedFiles)

    image_data = []
    label_data = []

    pos = 0

    for f in sortedFiles:
        print(str(pos) + ':  ' + f)
        pos += 1

        # * Generate image data
        img = Image.open(imgDir + '/' + f)
        img_arr = np.array(img)
        image_data.append(img_arr)

        # * Get label of image
        label = int((f.split('_')[0]))
        label_data.append(label)

    image_data_np = np.array(image_data)
    label_data_np = np.array(label_data)
    
    # * Save dataset as numpy arrays
    np.save(targetDir + r"\numbers_images.npy", image_data_np)
    np.save(targetDir + r"\numbers_labels.npy", label_data_np)

def generate_letter_dataset(imgDir, targetDir):
    '''Generates the dataset for letter glyphs.
    Both capital and lowercase letters are classified with the same label
    Also randomizes order of glyphs'''

    imgFiles = os.listdir(imgDir)
    sortedFiles = []

    #* Sort out glyph types to only incude letters
    for f in imgFiles:
        if f.split('_')[1] == 'upper' or f.split('_')[1] == 'lower':
            sortedFiles.append(f)

    #* Randomize glyphs
    random.shuffle(sortedFiles)

    image_data = []
    label_data = []

    pos = 0

    for f in sortedFiles:
        print(str(pos) + ':  ' + f)
        pos += 1

        #* Generate image data
        img = Image.open(imgDir + '/' + f)
        img_arr = np.array(img)
        image_data.append(img_arr)

        #* Get label of image
        #* Labels: 0 = a, 1 = b etc
        label = ord(f.split('_')[0]) - 96
        label_data.append(label)

    image_data_np = np.array(image_data)
    label_data_np = np.array(label_data)

    #* Save dataset as numpy arrays
    np.save(targetDir + r"\letters_images.npy", image_data_np)
    np.save(targetDir + r"\letters_labels.npy", label_data_np)
