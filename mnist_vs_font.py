from evaluation_cnn import EvalNetwork

import os
import tensorflow as tf
import numpy as np

from keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import csv


def format_data(data_images, data_labels):
    data_images = data_images.reshape((data_images.shape[0], 28, 28, 1))
    data_images = data_images.astype('float32') / 255
    if (np.amax(data_labels) == 26):
        data_labels = data_labels - 1
    data_labels = to_categorical(data_labels)
    return data_images, data_labels

def load_data(dataset):
    ''' Loads training and testing data samples from the given dataset.
    Valid datasets incude:
    - fonts_letters
    - fonts_numbers
    - emnist_letters
    - emnist_numbers
    - fonts_letters_no_val
    - fonts_numbers_no_val
    '''
    
    training_images = None
    training_labels = None
    testing_images = None
    testing_labels = None

    if (dataset == 'fonts_letters'):

        train_images = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/letters_images.npy")
        train_labels = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/letters_labels.npy")
        test_images, test_labels = extract_test_samples('letters')
        # test_images, test_labels = extract_training_samples('letters')

        training_images, training_labels = format_data(train_images, train_labels)
        testing_images, testing_labels = format_data(test_images, test_labels)

        print("Loaded fonts training letter dataset.")
        print("Loaded EMNIST testing letter dataset")

    if (dataset == 'fonts_numbers'):

        train_images = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/numbers_images.npy")
        train_labels = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/numbers_labels.npy")
        test_images, test_labels = extract_test_samples('digits')
        # test_images, test_labels = extract_training_samples('digits')

        training_images, training_labels = format_data(train_images, train_labels)
        testing_images, testing_labels = format_data(test_images, test_labels)

        print("Loaded fonts training number dataset.")
        print("Loaded EMNIST testing number dataset")

    if (dataset == 'emnist_letters'):

        train_images, train_labels = extract_training_samples('letters')
        test_images, test_labels = extract_test_samples('letters')

        training_images, training_labels = format_data(train_images, train_labels)
        testing_images, testing_labels = format_data(test_images, test_labels)

        print("Loaded EMNIST training letter dataset.")
        print("Loaded EMNIST testing letter dataset")

    if (dataset == 'emnist_numbers'):

        train_images, train_labels = extract_training_samples('digits')
        test_images, test_labels = extract_test_samples('digits')

        training_images, training_labels = format_data(train_images, train_labels)
        testing_images, testing_labels = format_data(test_images, test_labels)

        print("Loaded EMNIST training number dataset.")
        print("Loaded EMNIST testing number dataset")
        
    if (dataset == 'fonts_letters_no_val'):

        train_images = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/letters_images.npy")
        train_labels = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/letters_labels.npy")
        
        test_images = train_images[:14000]
        test_labels = train_labels[:14000]
        train_images = train_images[14000:]
        train_labels = train_labels[14000:]

        training_images, training_labels = format_data(train_images, train_labels)
        testing_images, testing_labels = format_data(test_images, test_labels)

        print("Loaded fonts training letter dataset.")
        #print("Loaded EMNIST testing letter dataset")
        
    if (dataset == 'fonts_numbers_no_val'):

        train_images = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/numbers_images.npy")
        train_labels = np.load("./font_dataset/_Format_v2/_optimizing_dataset/_datasets/numbers_labels.npy")
        
        test_images = train_images[:2700]
        test_labels = train_labels[:2700]
        train_images = train_images[2700:]
        train_labels = train_labels[2700:]

        training_images, training_labels = format_data(train_images, train_labels)
        testing_images, testing_labels = format_data(test_images, test_labels)

        print("Loaded fonts training letter dataset.")
        #print("Loaded EMNIST testing letter dataset")

    print('_________________________________________________________________')
    
    return ((training_images, training_labels), (testing_images, testing_labels))

def evaluate_dataset(dataset, n_epochs, n_batch_size, n_runs, csv_file=' '):

    data = load_data(dataset)
    
    training_images = data[0][0]
    training_labels = data[0][1]
    testing_images = data[1][0]
    testing_labels = data[1][1]
    
    network = None

    evaluation_data = []

    for i in range(n_runs):
        
        print("Run Nr " + str(i+1) + "...")
        print('_________________________________________________________________')

        network = EvalNetwork(dataset.split('_')[1])
        evaluation_data.append(network.train_network(

                (training_images, training_labels),
                (testing_images, testing_labels),
                n_epochs, n_batch_size
        ))
        #* If data is already located in csv file, they are opened from here
        with open(csv_file, mode='a') as eval_data:
            data_writer = csv.writer(eval_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(evaluation_data[-1].get('acc') + ["|"] + evaluation_data[-1].get('val_acc'))
        print('_________________________________________________________________')
        
    avr_acc     = [0] * n_epochs
    avr_val_acc = [0] * n_epochs
    
    #* Calculate average accuracys
    for i in range(n_runs):
        acc =     evaluation_data[i].get('acc')
        val_acc = evaluation_data[i].get('val_acc')
        
        for j in range(n_epochs):
            avr_acc[j]     += acc[j]     / n_runs
            avr_val_acc[j] += val_acc[j] / n_runs
            
    print(avr_acc)
    print(avr_val_acc)
        
    plt.figure(figsize=(8, 6))
    
    x = list(range(1, n_epochs + 1))
    plt.plot(x, avr_acc, '-b.', markerfacecolor='none', label="Font network accuracy")
    plt.plot(x, avr_val_acc, '-gx', label="EMNIST validation accuracy")
    
    #plt.xticks([0, 5, 10, 15, 20, 25, 30])
    plt.title("Font Dataset Performance validated with EMNIST Dataset \n- " + dataset.split('_')[1].upper() + " -", fontsize=15, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    
    
def generate_confusion_matrix(dataset, n_epochs, n_batch_size):
    
    #* Generate confusion matrix
    #* 1st of validation set
    #* 2nd of training set
        
    data = load_data(dataset)
        
    training_images = data[0][0]
    training_labels = data[0][1]
    validation_images = data[1][0]
    validation_labels = data[1][1]
        
    network = EvalNetwork(dataset.split('_')[1])
        


    network.train_network((training_images, training_labels),
                          (validation_images, validation_labels),
                          n_epochs, n_batch_size)
        
    val_predictions = network.model.predict(validation_images)
    train_predictions = network.model.predict(training_images)
    
    val_true_labels      = [i.index(max(i)) for i in validation_labels.tolist()]
    val_predicted_labels = [i.index(max(i)) for i in val_predictions.tolist()]
    
    train_true_labels      = [i.index(max(i)) for i in training_labels.tolist()]
    train_predicted_labels = [i.index(max(i)) for i in train_predictions.tolist()]
    
    if (dataset == 'fonts_letters' or dataset == 'emnist_letters'):
        val_confusion_matrix   = np.zeros((26,26), dtype=int).tolist()
        train_confusion_matrix = np.zeros((26,26), dtype=int).tolist()
    else:
        val_confusion_matrix   = np.zeros((10,10), dtype=int).tolist()
        train_confusion_matrix = np.zeros((10,10), dtype=int).tolist()

    for i in range(len(val_true_labels)):
        val_confusion_matrix[val_true_labels[i]][val_predicted_labels[i]] += 1

    for i in range(len(train_true_labels)):
        train_confusion_matrix[train_true_labels[i]][train_predicted_labels[i]] += 1
        
    val_df_cm   = pd.DataFrame(val_confusion_matrix)
    train_df_cm = pd.DataFrame(train_confusion_matrix)
    
    if (dataset == 'fonts_letters' or dataset == 'emnist_letters'):
        tick_label = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    else:
        tick_label = [i for i in "0123456789"]
        
    plt.figure(figsize=(24, 16))
    sn.set(font_scale=1.4)
    sn.heatmap(val_df_cm, annot=True, annot_kws={"size": 16}, fmt='d', linewidths=1, cmap=sn.cm.rocket_r,
               xticklabels=tick_label,
               yticklabels=tick_label)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion matrix \n- " + "NUMBERS" + " -", fontsize=20, fontweight="bold")
    plt.show()
    
    plt.figure(figsize=(24, 16))
    sn.set(font_scale=1.4)
    sn.heatmap(train_df_cm, annot=True, annot_kws={"size": 16}, fmt='d', linewidths=1, cmap=sn.cm.rocket_r,
               xticklabels=tick_label,
               yticklabels=tick_label)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion matrix \n- " + "NUMBERS" + " -", fontsize=20, fontweight="bold")
    plt.show()
