import numpy as np, argparse, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, PReLU, Dense, BatchNormalization, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

def main(args, train_X, train_y):
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
    model.fit(train_X,train_y,batch_size=16,epochs=args.epochs)
    save_model_path = os.path.join(args.sm_model_dir, '000000001')
    model.save(save_model_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--epochs', type=int, default=1)

    return parser.parse_known_args()

def load_training_data(base_dir):
    """Load MNIST training data"""
    print(os.path.join(base_dir, 'train_X.npy'))
    train_X = np.load(os.path.join(base_dir, 'train_X.npy'))
    train_y = np.load(os.path.join(base_dir, 'train_y.npy'))
    return train_X, train_y
    
if __name__=='__main__':
    argv = argparse.ArgumentParser()
    args, unknown = parse_args()
    print(args.train)
    print(os.listdir(args.train))
    train_X, train_y = load_training_data(args.train)
    main(args, train_X, train_y)