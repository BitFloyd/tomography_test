import matplotlib as mpl
mpl.use('Agg')
from skimage import color
from skimage.io import imread
from sklearn.feature_extraction.image import extract_patches
from skimage.filters import gaussian
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Conv2D,GaussianNoise,SpatialDropout2D,LeakyReLU
from keras.layers import BatchNormalization,AveragePooling2D,Dense,Flatten
from keras.layers import Input,Dropout,Reshape,UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras_contrib.losses import DSSIMObjective

from tqdm import tqdm
import os


def message_print(string_h,string_val):
    print "###########################################"
    print string_h, ':',string_val
    print "###########################################"

    return True

def create_autoencoder(size_y=128,size_x=128,n_channels=1,h_units=16):

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
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 64x64

    x1 = GaussianNoise(0.03)(x1)
    x1 = Conv2D(f2, (3, 3), padding='same')(x1)
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

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(f4, (3, 3), padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)                # 8x8

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(units=h_units)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    encoder = BatchNormalization()(x1)

    dec1 = Dense(units=(size_y / 16) * (size_x / 16) * f4)
    dec2 = LeakyReLU(alpha=0.2)
    dec3 = Reshape((size_x / 16, size_y / 16, f4))

    dec4 = UpSampling2D(size=(2, 2))
    dec5 = Conv2D(f5, (3, 3), padding='same')
    dec6 = LeakyReLU(alpha=0.2)
    dec7 = BatchNormalization()  # 16x16

    dec8 = UpSampling2D(size=(2, 2))
    dec9 = Conv2D(f6, (3, 3), padding='same')
    dec10 = LeakyReLU(alpha=0.2)
    dec11 = BatchNormalization()  # 32x32

    dec12 = UpSampling2D(size=(2, 2))
    dec13 = Conv2D(f7, (3, 3), padding='same')
    dec14 = LeakyReLU(alpha=0.2)
    dec15 = BatchNormalization()  # 64x64

    dec16 = UpSampling2D(size=(2, 2))
    dec17 = Conv2D(f8, (3, 3), padding='same')
    dec18 = LeakyReLU(alpha=0.2)
    dec19 = BatchNormalization()  # 128x128

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
    ae20= recon(ae19)

    adam_ae = Adam(lr=1e-3)

    ae = Model(inputs=[inp], outputs=[ae20])

    dssim = DSSIMObjective(kernel_size=4)

    ae.compile(optimizer=adam_ae, loss=dssim)

    return ae

def extract_image_patches_gs(input_image,patch_shape=(32,32),strides=(2,2)):

    y_pos = 0
    x_pos = 0

    y_length = input_image.shape[0]
    x_length = input_image.shape[1]

    strides_y = strides[0]
    strides_x = strides[1]

    list_patches = []

    while y_pos+patch_shape[0] < y_length:

        x_pos = 0

        while x_pos+patch_shape[1] < x_length:

            patch = input_image[y_pos:y_pos+patch_shape[0],x_pos:x_pos+patch_shape[1]]
            list_patches.append(patch)
            x_pos+=strides_x

        y_pos+=strides_y


    return list_patches

def return_train_test_patches(input_path,output_path,size_y,size_x):

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
        output_image_i = gaussian(output_image_i,sigma=0.4)

        input_image_patches = extract_image_patches_gs(input_image_i,patch_shape=(size_y,size_x),strides=(size_y/4,size_x/4))
        output_image_patches = extract_image_patches_gs(output_image_i,patch_shape=(size_y,size_x),strides=(size_y/4,size_x/4))

        list_input_images.extend(input_image_patches)
        list_output_images.extend(output_image_patches)


    message_print("NUM_INP_IMAGE_PATCHES",str(len(list_input_images)))
    message_print("NUM_OUTP_IMAGE_PATCHES", str(len(list_output_images)))

    list_input_images = np.array(list_input_images)
    list_output_images = np.array(list_output_images)

    list_input_images = np.expand_dims(list_input_images,-1)
    list_output_images = np.expand_dims(list_output_images,-1)

    return list_input_images, list_output_images

def create_plots(X, Y, n=10, filename='plot.png'):

    for i in range(0,n):
        fig, axes = plt.subplots(1,figsize=(20, 20))

        x_img = X[i].reshape(size_y,size_x)
        x_img = np.dstack([x_img,x_img,x_img])

        color_mask = np.zeros((size_y,size_x,3))
        color_mask[(Y[i].reshape(size_x,size_y)>0.1)] = [1,0,0]

        x_img_hsv = color.rgb2hsv(x_img)
        color_mask_hsv = color.rgb2hsv(color_mask)

        x_img_hsv[..., 0] = color_mask_hsv[..., 0]
        x_img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.9

        x_img = color.hsv2rgb(x_img_hsv)

        axes.imshow(x_img,interpolation='nearest')
        axes.get_xaxis().set_ticks([])
        axes.get_yaxis().set_ticks([])
        plt.savefig(os.path.join('data',filename+'_'+str(i)+'.png'),bbox_inches='tight')
        plt.close()

    return True

if __name__== "__main__":

    input_path = os.path.join('data','Input')
    output_path = os.path.join('data','Output')

    size_x = 128
    size_y = 128

    X, Y = return_train_test_patches(input_path,output_path,size_y,size_x)

    ae = create_autoencoder(size_y,size_x,n_channels=1,h_units=16)

    ae.summary()

    Y = np.float32(Y>0.1)

    X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,shuffle=True)

    create_plots(X_train, Y_train, n=50, filename='Train/train_imgs_plot')

    es = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    mcp = ModelCheckpoint(filepath='model_saved.h5',monitor='val_loss',verbose=1,save_best_only=True)

    print ("START MODEL FITTING")
    # fits the model on batches with real-time data augmentation:
    ae.fit(X_train,Y_train,epochs=50, callbacks=[es, rlr,mcp],batch_size = 128, validation_data=(X_test,Y_test),shuffle=True)

    Y_pred = ae.predict(X_test)

    print np.max(Y_pred)

    create_plots(X_test, Y_pred, n=50, filename='Test/predicted_imgs_plot')
