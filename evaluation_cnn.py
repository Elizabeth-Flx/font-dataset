import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import layers
from keras import models

import matplotlib.pyplot as plt

class EvalNetwork:

    def __init__(self, output_type):

        #* Network Model
        self.model = models.Sequential()

        #* Convolutional Layers
        self.model.add(layers.Conv2D(64,  (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())

        #* Dense Layers
        self.model.add(layers.Dense(1600, activation='relu'))

        #* Output Layer
        if output_type == "numbers":
            self.model.add(layers.Dense(10, activation='softmax', name="output"))
        if output_type == "letters":
            self.model.add(layers.Dense(26, activation='softmax', name="output"))
            
        print("Loaded new evaluation model for " + output_type + ".")
        print('_________________________________________________________________')

        #* Quick model summary
        #self.model.summary()

    def train_network(self, training_data, testing_data, n_epochs, n_batch_size, visualize_training=False):

        self.model.compile( optimizer=  'adam',
                            loss=       'categorical_crossentropy',
                            metrics=    ['accuracy'])

        training_hist = self.model.fit(training_data[0], training_data[1],
                                        epochs=n_epochs, batch_size=n_batch_size,
                                        validation_data=testing_data,
                                        verbose=2)

        #* If visualize_training is True, then show the results using matplotlib
        if visualize_training:
            x = list(range(1, n_epochs + 1))
            plt.plot(x, training_hist.history.get('acc'))
            plt.plot(x, training_hist.history.get('val_acc'))
            plt.show()

        return training_hist.history
    
    
    def train_network_no_validation_data(self, training_data, validation_split, n_epochs, n_batch_size, visualize_training=False):
    
        self.model.compile( optimizer=  'adam',
                            loss=       'categorical_crossentropy',
                            metrics=    ['accuracy'])
        
        training_hist = self.model.fit(training_data[0], training_data[1],
                                        epochs=n_epochs, batch_size=n_batch_size,
                                        validation_split=validation_split,
                                        verbose=2)
        
        #* If visualize_training is True, then show the results using matplotlib
        if visualize_training:
            x = list(range(1, n_epochs + 1))
            plt.plot(x, training_hist.history.get('acc'))
            plt.plot(x, training_hist.history.get('val_acc'))
            plt.show()
