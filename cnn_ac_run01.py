from skimage.io import imread
from skimage.transform import resize as imresize
import numpy as np
from sklearn.model_selection import train_test_split


from keras.models import Model
from keras.layers import Conv2D,GaussianNoise,SpatialDropout2D,LeakyReLU
from keras.layers import BatchNormalization,AveragePooling2D,Dense,Flatten
from keras.layers import Input,Dropout,Reshape,UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tqdm import tqdm
import os


def create_autoencoder(size_y=256,size_x=256,n_channels=3,h_units=256):

    f1 = 64
    f2 = 128
    f3 = 256
    f4 = 512
    f5 = 256
    f6 = 128
    f7 = 64
    f8 = 32


    # MODEL CREATION
    inp = Input(shape=(size_y, size_x, n_channels))
    x1 = GaussianNoise(0.05)(inp)
    x1 = Conv2D(f1, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 16x16

    x1 = GaussianNoise(0.03)(x1)
    x1 = Conv2D(f2, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 8x8

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 4x4

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(units=h_units)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    encoder = BatchNormalization()(x1)

    dec1 = Dense(units=(size_y / 8) * (size_x / 8) * f4)
    dec2 = LeakyReLU(alpha=0.2)
    dec3 = Reshape((size_x / 8, size_y / 8, f4))

    dec4 = UpSampling2D(size=(2, 2))
    dec5 = Conv2D(f5, (3, 3), padding='same')
    dec6 = LeakyReLU(alpha=0.2)
    dec7 = BatchNormalization()  # 8x8

    dec8 = UpSampling2D(size=(2, 2))
    dec9 = Conv2D(f6, (3, 3), padding='same')
    dec10 = LeakyReLU(alpha=0.2)
    dec11 = BatchNormalization()  # 16x16

    dec12 = UpSampling2D(size=(2, 2))
    dec13 = Conv2D(f7, (3, 3), padding='same')
    dec14 = LeakyReLU(alpha=0.2)
    dec15 = BatchNormalization()  # 32x32

    dec16 = Conv2D(f8, (3, 3), padding='same')
    dec17 = LeakyReLU(alpha=0.2)  # 32x32

    recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

    ae1 = dec1(encoder)
    ae2 = dec2(ae1)
    ae3 = dec3(ae2)
    ae4 = dec4(ae3)
    ae5 = dec5(ae4)
    ae6 = dec6(ae5)
    ae7 = dec7(ae6)
    ae8 = dec8(ae7)
    ae9 = dec9(ae8)
    ae10 = dec10(ae9)
    ae11 = dec11(ae10)
    ae12 = dec12(ae11)
    ae13 = dec13(ae12)
    ae14 = dec14(ae13)
    ae15 = dec15(ae14)
    ae16 = dec16(ae15)
    ae17 = dec17(ae16)
    ae18 = recon(ae17)

    adam_ae = Adam(lr=1e-3)

    ae = Model(inputs=[inp], outputs=[ae18])

    ae.compile(optimizer=adam_ae, loss='binary_crossentropy')

    return ae

def return_train_test_files(input_path,output_path,size_y,size_x):

    list_input_files = os.listdir(input_path)
    list_input_files.sort()
    list_output_files = os.listdir(output_path)
    list_output_files.sort()

    list_input_images = []
    list_output_images = []

    for i in tqdm(range(1,len(list_input_files)+1)):

        fname_input = os.path.join(input_path,'jtf_'+str(i).zfill(3)+'.tif')
        fname_output = os.path.join(output_path,'jtf'+str(i)+'_bin.tiff')

        input_image_i = imread(fname=fname_input)
        output_image_i = imread(fname=fname_output)
        input_image_i = imresize(input_image_i,(size_y,size_x))
        output_image_i = imresize(output_image_i,(size_y,size_x))

        input_image_i = np.expand_dims(input_image_i,axis=-1)
        output_image_i = np.expand_dims(output_image_i,axis=-1)

        list_input_images.append(input_image_i)
        list_output_images.append(output_image_i)


    return np.array(list_input_images), np.array(list_output_images)


if __name__== "__main__":

    input_path = os.path.join('data','Input')
    output_path = os.path.join('data','Output')

    size_x = 256
    size_y = 256

    X, Y = return_train_test_files(input_path,output_path,size_y,size_x)

    ae = create_autoencoder(size_y,size_x,n_channels=1,h_units=128)

    X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath='model_saved.h5',monitor='val_loss',verbose=1,save_best_only=True)

    print ("START MODEL FITTING")
    # fits the model on batches with real-time data augmentation:
    ae.fit(X_train,Y_train,epochs=200, callbacks=[es, rlr,mcp],batch_size = 32,
              validation_data=(X_test,Y_test))