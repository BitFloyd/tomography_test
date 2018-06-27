from skimage.io import imread
from skimage.transform import resize as imresize
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 1024x1024

    x1 = GaussianNoise(0.03)(x1)
    x1 = Conv2D(f2, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 512x512

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 256x256

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 128x128

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 64x64

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 32x32

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 16x16

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f3, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 8x8

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(units=h_units)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    encoder = BatchNormalization()(x1)

    dec1 = Dense(units=(size_y / 256) * (size_x / 256) * f4)
    dec2 = LeakyReLU(alpha=0.2)
    dec3 = Reshape((size_x / 256, size_y / 256, f4))

    dec4 = UpSampling2D(size=(2, 2))
    dec5 = Conv2D(f5, (3, 3), padding='same')
    dec6 = LeakyReLU(alpha=0.2)
    dec7 = BatchNormalization()  # 16x16

    dec8 = UpSampling2D(size=(2, 2))
    dec9 = Conv2D(f5, (3, 3), padding='same')
    dec10 = LeakyReLU(alpha=0.2)
    dec11 = BatchNormalization()  # 32x32

    dec12 = UpSampling2D(size=(2, 2))
    dec13 = Conv2D(f5, (3, 3), padding='same')
    dec14 = LeakyReLU(alpha=0.2)
    dec15 = BatchNormalization()  # 64x64

    dec16 = UpSampling2D(size=(2, 2))
    dec17 = Conv2D(f5, (3, 3), padding='same')
    dec18 = LeakyReLU(alpha=0.2)
    dec19 = BatchNormalization()  # 128x128

    dec20 = UpSampling2D(size=(2, 2))
    dec21 = Conv2D(f5, (3, 3), padding='same')
    dec22 = LeakyReLU(alpha=0.2)
    dec23 = BatchNormalization()  # 256x256

    dec24 = UpSampling2D(size=(2, 2))
    dec25 = Conv2D(f5, (3, 3), padding='same')
    dec26 = LeakyReLU(alpha=0.2)
    dec27 = BatchNormalization()  # 512x512


    dec28 = UpSampling2D(size=(2, 2))
    dec29 = Conv2D(f6, (3, 3), padding='same')
    dec30 = LeakyReLU(alpha=0.2)
    dec31 = BatchNormalization()  # 1024x1024

    dec32 = UpSampling2D(size=(2, 2))
    dec33 = Conv2D(f7, (3, 3), padding='same')
    dec34 = LeakyReLU(alpha=0.2)
    dec35 = BatchNormalization()  # 2048x2048

    dec36 = Conv2D(f8, (3, 3), padding='same')
    dec37 = LeakyReLU(alpha=0.2)  # 2048x2048

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
    ae18 = dec18(ae17)
    ae19 = dec19(ae18)
    ae20 = dec20(ae19)
    ae21 = dec21(ae20)
    ae22 = dec22(ae21)
    ae23 = dec23(ae22)
    ae24 = dec24(ae23)
    ae25 = dec25(ae24)
    ae26 = dec26(ae25)
    ae27 = dec27(ae26)
    ae28 = dec28(ae27)
    ae29 = dec29(ae28)
    ae30 = dec30(ae29)
    ae31 = dec31(ae30)
    ae32 = dec32(ae31)
    ae33 = dec33(ae32)
    ae34 = dec34(ae33)
    ae35 = dec35(ae34)
    ae36 = dec36(ae35)
    ae37 = dec37(ae36)
    ae38 = recon(ae37)



    adam_ae = Adam(lr=1e-3)

    ae = Model(inputs=[inp], outputs=[ae38])

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

def create_plots(X, Y, n=10, filename='plot.png'):

    fig, axes = plt.subplots(n,2,figsize=(10,40))

    for i in range(0,n):

        axes[i][0].imshow(X[i].reshape(size_x,size_y),cmap=plt.get_cmap('gist_gray'))
        axes[i][0].get_xaxis().set_ticks([])
        axes[i][0].get_yaxis().set_ticks([])

        axes[i][1].imshow(Y[i].reshape(size_x,size_y),cmap=plt.get_cmap('gist_gray'))
        axes[i][1].get_xaxis().set_ticks([])
        axes[i][1].get_yaxis().set_ticks([])

    plt.savefig(os.path.join('data',filename),bbox_inches='tight')
    plt.close()
    return True

if __name__== "__main__":

    input_path = os.path.join('data','Input')
    output_path = os.path.join('data','Output')

    size_x = 2048
    size_y = 2048

    X, Y = return_train_test_files(input_path,output_path,size_y,size_x)

    Y = (Y>=0.5)*1.0

    ae = create_autoencoder(size_y,size_x,n_channels=1,h_units=32)

    ae.summary()

    X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    create_plots(X_test, Y_test, n=5, filename='train_imgs_plot.png')

    es = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    mcp = ModelCheckpoint(filepath='model_saved.h5',monitor='val_loss',verbose=1,save_best_only=True)

    print ("START MODEL FITTING")
    # fits the model on batches with real-time data augmentation:
    ae.fit(X_train,Y_train,epochs=200, callbacks=[es, rlr,mcp],batch_size = 256,
              validation_data=(X_test,Y_test))

    Y_pred = ae.predict(X_test)

    create_plots(X_test, Y_test, n=5, filename='predicted_imgs_plot.png')
