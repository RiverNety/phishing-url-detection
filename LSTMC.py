from string import printable
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Embedding,
    BatchNormalization, LSTM
)
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K
from utils import load_model, save_model
from tensorflow.keras.utils import plot_model


class LSTMC:

    def __init__(self, max_len=75, emb_dim=32, max_vocab_len=100, w_reg=tf.keras.regularizers.l2(1e-4)):
        self.max_len = max_len
        self.csv_logger = CSVLogger('LSTM_log.csv', append=True, separator=';')

        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(
            input_dim=max_vocab_len,
            output_dim=emb_dim,
            input_length=max_len,
            embeddings_regularizer=w_reg
        )(main_input)
        emb = Dropout(0.25)(emb)

        # LSTM layer
        lstm_out = LSTM(256, return_sequences=False)(emb)
        lstm_out = Dropout(0.5)(lstm_out)

        # Dense layer 1
        hidden1 = Dense(512)(lstm_out)
        hidden1 = tf.keras.layers.ELU()(hidden1)
        hidden1 = BatchNormalization()(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        # Dense layer 2
        hidden2 = Dense(512)(hidden1)
        hidden2 = tf.keras.layers.ELU()(hidden2)
        hidden2 = BatchNormalization()(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(hidden2)

        # Compile
        self.model = Model(inputs=[main_input], outputs=[output])
        self.adam = Adam(learning_rate=1e-4)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])


    def save_model(self, fileModelJSON, fileWeights):
        save_model(self.model, fileModelJSON, fileWeights)

    def load_model(self, fileModelJSON, fileWeights):
        self.model = load_model(fileModelJSON, fileWeights)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, target_train, epochs=5, batch_size=32):
        print(f"Training LSTM model with {epochs} epochs and batch size {batch_size}")
        self.model.fit(
            x_train, target_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[self.csv_logger]
        )

    def test_model(self, x_test, target_test):
        print("Testing LSTM model")
        return self.model.evaluate(x_test, target_test, verbose=1)

    def predict(self, x_input):
        url_int_tokens = [[printable.index(x) + 1 for x in x_input if x in printable]]
        X = sequence.pad_sequences(url_int_tokens, maxlen=self.max_len)
        p = self.model.predict(X, batch_size=1)
        return "benign" if p < 0.5 else "malicious"

    def export_plot(self):
        plot_model(self.model, to_file='LSTM.png', show_shapes=True)
