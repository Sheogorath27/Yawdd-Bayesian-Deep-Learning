import keras
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, LSTM
from keras import backend as K

from concreteLayers import SpatialConcreteDropout, ConcreteDropout


def lenet(input_shape, num_classes, optimizer = keras.optimizers.SGD()):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=20, kernel_size=5, strides=1)(inp)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=50, kernel_size=5, strides=1)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model =  Model(inp, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def lenetDrp(input_shape, num_classes, optimizer = keras.optimizers.SGD()):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=20, kernel_size=5, strides=1, seed = 234)(inp)
    x = Dropout(0.01)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=50, kernel_size=5, strides=1)(x)
    x = Dropout(0.1)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu',seed = 1213)(x)
    x = Dropout(0.4, seed=123454)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model =  Model(inp, x)

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy']) 
    return model

def lenetMcDrp(input_shape, num_classes, optimizer = keras.optimizers.SGD()):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=20, kernel_size=5, strides=1)(inp)
    x = Dropout(0.1, seed = 132424)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(filters=50, kernel_size=5, strides=1)(x)
    x = Dropout(0.1, seed = 90764)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.4, seed = 1324)(x, training=True)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def lenetCncDrp(input_shape, num_classes, wd, dd, optimizer = keras.optimizers.SGD()):
  model = Sequential()
  model.add(SpatialConcreteDropout(Conv2D(20, kernel_size=(5, 5),
                                        activation='relu'),
                                 weight_regularizer=wd, dropout_regularizer=dd,
                                 init_min=0.001, init_max=0.01,
                                 input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(SpatialConcreteDropout(Conv2D(50, kernel_size=(5, 5),
                                        activation='relu'),
                                 weight_regularizer=wd, dropout_regularizer=dd,
                                 init_min=0.001, init_max=0.01,
                                 input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(Flatten())
  model.add(ConcreteDropout(Dense(500, activation='relu'), 
                          weight_regularizer=wd, dropout_regularizer=dd,
                          init_min=0.01, init_max=0.1))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])  
  return model

  def lenetLSTM(input_shape, num_classes, optimizer = keras.optimizers.SGD()):
    inp = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(filters=20, kernel_size=5, strides=1))(inp)
    x = TimeDistributed(MaxPool2D(pool_size=2, strides=2))(x)
    x = TimeDistributed(Conv2D(filters=50, kernel_size=5, strides=1))(x)
    x = TimeDistributed(MaxPool2D(pool_size=2, strides=2))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1000, activation='relu'))(x)
    x = LSTM(5000)(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model =  Model(inp, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model