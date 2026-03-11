import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import Huber

class GRUModel:
    def __init__(self, input_width, num_hidden_layer, num_features, output_length):
        self.input_width = input_width
        self.num_hidden_layer = num_hidden_layer
        self.num_features = num_features
        self.output_length = output_length
        self.model = self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=(self.input_width, self.num_features))
        x = input_layer
        
        for _ in range(self.num_hidden_layer):
            x = layers.GRU(units=128, return_sequences=True)(x)
            
        x = layers.GRU(units=128, return_sequences=False)(x)
        x = layers.Dense(units=self.output_length)(x)
        
        model = Model(inputs=input_layer, outputs=x)
        return model

    def compile_model(self, optimizer, metrics):
        self.model.compile(loss=Huber(),
                           optimizer=optimizer,
                           metrics=metrics)

class TCNModel:
    def __init__(self, input_width, num_features, output_length, nb_filters=64, kernel_size=3, nb_stacks=1, dilations=[1, 2, 4, 8, 16, 32], **kwargs):
        self.input_width = input_width
        self.num_features = num_features
        self.output_length = output_length
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations
        self.model = self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=(self.input_width, self.num_features))
        x = input_layer
        
        for s in range(self.nb_stacks):
            for d in self.dilations:
                # Residual block
                prev_x = x
                
                # Dilated Conv 1
                x = layers.Conv1D(filters=self.nb_filters, 
                                  kernel_size=self.kernel_size, 
                                  dilation_rate=d, 
                                  padding='causal',
                                  activation='relu')(x)
                x = layers.Dropout(0.1)(x)
                
                # Dilated Conv 2
                x = layers.Conv1D(filters=self.nb_filters, 
                                  kernel_size=self.kernel_size, 
                                  dilation_rate=d, 
                                  padding='causal',
                                  activation='relu')(x)
                x = layers.Dropout(0.1)(x)
                
                # Skip connection
                if prev_x.shape[-1] != x.shape[-1]:
                    prev_x = layers.Conv1D(filters=self.nb_filters, kernel_size=1, padding='same')(prev_x)
                
                x = layers.Add()([prev_x, x])
        
        # Output layer
        x = layers.Lambda(lambda tt: tt[:, -1, :])(x)
        x = layers.Dense(units=128, activation='relu')(x)
        x = layers.Dense(units=self.output_length)(x)
        
        model = Model(inputs=input_layer, outputs=x)
        return model

    def compile_model(self, optimizer, metrics):
        self.model.compile(loss=Huber(),
                           optimizer=optimizer,
                           metrics=metrics)
