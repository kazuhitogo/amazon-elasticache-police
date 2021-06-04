import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, PReLU, Dense, BatchNormalization, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

def main():
    train_X = np.load('./train_X.npy')
    train_y = np.load('./train_y.npy')
    inputs = Input(shape=(50,700,1))
    x = Conv2D(64, (3,3),padding='same')(inputs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'],loss="binary_crossentropy")
    model.fit(train_X,train_y,batch_size=16,epochs=10)
    model.save('elasticache_classifier.h5')

if __name__=='__main__':
    main()