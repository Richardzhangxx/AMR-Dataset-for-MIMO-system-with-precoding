import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.io as scio
import csv
import keras
import numpy as np
import pandas as pd
import LSTM as mcl
import mltools
import matplotlib.pyplot as plt
import pickle
# data concatenate
filename=r'./Nt4Nr2'
dataFile1 = filename+r'/data2psk.mat'
data1 = scio.loadmat(dataFile1)

dataFile2 = filename+r'/dataqpsk.mat'
data2 = scio.loadmat(dataFile2)

dataFile3 = filename+r'/data8psk.mat'
data3 = scio.loadmat(dataFile3)

dataFile4 = filename+r'/data16qam.mat'
data4 = scio.loadmat(dataFile4)

dataFile5 = filename+r'/data64qam.mat'
data5 = scio.loadmat(dataFile5)

dataFile6 = filename+r'/data128qam.mat'
data6 = scio.loadmat(dataFile6)

# label concatenate
dataFile11 = filename+r'/label2psk.mat'
label1 = scio.loadmat(dataFile11)

dataFile22 = filename+r'/labelqpsk.mat'
label2 = scio.loadmat(dataFile22)

dataFile33 = filename+r'/label8psk.mat'
label3 = scio.loadmat(dataFile33)

dataFile44 = filename+r'/label16qam.mat'
label4 = scio.loadmat(dataFile44)

dataFile55 = filename+r'/label64qam.mat'
label5 = scio.loadmat(dataFile55)

dataFile66 = filename+r'/label128qam.mat'
label6 = scio.loadmat(dataFile66)
# snr concatenate
dataFile111 = filename+r'/snr2psk.mat'
snr1 = scio.loadmat(dataFile111)

dataFile222 = filename+r'/snrqpsk.mat'
snr2 = scio.loadmat(dataFile222)

dataFile333 = filename+r'/snr8psk.mat'
snr3 = scio.loadmat(dataFile333)

dataFile444 = filename+r'/snr16qam.mat'
snr4 = scio.loadmat(dataFile444)

dataFile555 =filename+r'/snr64qam.mat'
snr5 = scio.loadmat(dataFile555)

dataFile666 = filename+r'/snr128qam.mat'
snr6 = scio.loadmat(dataFile666)

dataset=np.concatenate([data1['data_save'],data2['data_save'],data3['data_save'],data4['data_save'],data5['data_save'],data6['data_save']])
label=np.concatenate([label1['label_save'],label2['label_save'],label3['label_save'],label4['label_save'],label5['label_save'],label6['label_save']])
snrs=np.concatenate([snr1['snr_save'],snr2['snr_save'],snr3['snr_save'],snr4['snr_save'],snr5['snr_save'],snr6['snr_save']])

L=500
snr_num=31
a=0
train_idx=[]
val_idx=[]
n_examples=dataset.shape[0]
num_mod=6
for j in range(num_mod):
    for i in range(snr_num):
        train_idx+=list(np.random.choice(range(a*L,(a+1)*L),size=300,replace=False))
        val_idx+=list(np.random.choice(list(set(range(a*L,(a+1)*L))-set(train_idx)), size=100, replace=False))
        a+=1
test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))

np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)

X_train = dataset[train_idx]
X_val=dataset[val_idx]
X_test =  dataset[test_idx]
X_train=np.expand_dims(X_train,axis=3)
X_test=np.expand_dims(X_test,axis=3)
X_val=np.expand_dims(X_val,axis=3)
Y_train = label[train_idx]
Y_val=label[val_idx]
Y_test = label[test_idx]
batch_size= 128
nb_epoch = 1000
model=mcl.LSTM()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()
filepath = 'weights.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val,Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patince=5,min_lr=0.0000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                ])
mltools.show_history(history)
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)
classes=['2PSK', 'QPSK', '8PSK', '16QAM', '64QAM', '128QAM']
def predict(model):
    model.load_weights(filepath)
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, 6)
    mltools.plot_confusion_matrix(confnorm,
                                  labels=['2PSK', 'QPSK', '8PSK', '16QAM', '64QAM', '128QAM'], save_filename='figure/mclstm_total_confusion.png')
    acc = {}
    acc_mod_snr = np.zeros((6,31))
    i = 0
    for snr in range(-10,21):
        # Extract classes @ SNR
        test_SNRs = snrs[test_idx].reshape(len(test_idx))
        test_X_i = X_test[np.where(test_SNRs == snr)]
        test_Y_i = Y_test[np.where(test_SNRs == snr)]
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, 6)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        mltools.plot_confusion_matrix(confnorm_i,
                                      labels=['2PSK', 'QPSK', '8PSK', '16QAM', '64QAM', '128QAM'], title="Confusion Matrix",
                                      save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr, 100.0 * acc[snr]))
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i = i + 1
predict(model)
