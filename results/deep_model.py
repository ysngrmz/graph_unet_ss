import sys

sys.path.insert(0, "../scripts/")
sys.path.insert(0, "../scripts/layers_gcn/")

from keras.models import Model, Input
from keras.layers import concatenate, LSTM, Dense, Activation, Dropout, Conv1D, BatchNormalization, Bidirectional

import keras
from tensorflow.keras import layers
import tensorflow as tf

import numpy as np
from keras.optimizers import Adam

from keras import initializers

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

from utils import *
from multi_graph_cnn_layer import MultiGraphCNN

from gends import GenerateDs
ds = GenerateDs()

import generate_n_window_dataset_for_fullpredict_for_given_path as gd
import performance_metrics as pm

acv_fnc = "relu"

ws = sys.argv[1]
nlabel = sys.argv[2]
tr_dir = sys.argv[3]
te_dir = sys.argv[4]
val_dir = sys.argv[5]
dsName = sys.argv[6]

ws = int(ws)
nlabel = int(nlabel)

trainInputs,trainTargets = gd.generate_dataset(ws,nlabel,tr_dir)
testInputs,testTargets = gd.generate_dataset(ws,nlabel,te_dir)
valInputs,valTargets = gd.generate_dataset(ws,nlabel,val_dir)

testTargets_ = testTargets
trainTargets_ = trainTargets
valTargets_ = valTargets

trainTargets = np.reshape(trainTargets,(-1,ws,1))
testTargets = np.reshape(testTargets,(-1,ws,1))
valTargets = np.reshape(valTargets,(-1,ws,1))

number_of_tr_sample = trainInputs.shape[0]
number_of_te_sample = testInputs.shape[0]
number_of_vl_sample = valInputs.shape[0]
number_of_feature = trainInputs.shape[2]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, initializer, dropout=0):
	x = layers.LayerNormalization(epsilon=1e-6)(inputs)
	x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, kernel_initializer=initializer, dropout=dropout)(x, x)
	x = layers.Dropout(dropout)(x)
	res = x + inputs

	x = layers.LayerNormalization(epsilon=1e-6)(res)
	x = layers.Conv1D(filters=ff_dim, kernel_size=1, kernel_initializer=initializer, activation=acv_fnc)(x)
	x = layers.Dropout(dropout)(x)
	x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer=initializer)(x)
	return x + res

def CNNModule(inp,n_unit,initializer):
	conv7 = Conv1D(n_unit, kernel_size=7,padding='same',kernel_initializer=initializer)(inp)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation(acv_fnc)(conv7)

	conv7 = Conv1D(n_unit, kernel_size=7,padding='same',kernel_initializer=initializer)(conv7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation(acv_fnc)(conv7)

	conv7 = Conv1D(n_unit, kernel_size=7,padding='same',kernel_initializer=initializer)(conv7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation(acv_fnc)(conv7)

	return conv7

def CNNModule_2(inp,n_unit,initializer):
	conv7 = Conv1D(n_unit, kernel_size=7,padding='same',kernel_initializer=initializer)(inp)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation(acv_fnc)(conv7)

	conv7 = Conv1D(n_unit, kernel_size=7,padding='same',kernel_initializer=initializer)(conv7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation(acv_fnc)(conv7)

	return conv7



def GCNModule(inp,dim,n_filter,dr_rate,graph_conv_filters_input):
	gcn1 = MultiGraphCNN(dim, n_filter, activation='relu')([inp, graph_conv_filters_input])
	gcn1 = Dropout(dr_rate)(gcn1)
	return gcn1

def LSTMModule(inp,n_unit_lstm,r_seq,dr_rate):
	lstm_l1 = Bidirectional(LSTM(n_unit_lstm,return_sequences=r_seq))(inp)
	lstm_l1 = BatchNormalization()(lstm_l1)
	lstm_l1 = Activation("relu")(lstm_l1)
	lstm_l1 = Dropout(dr_rate)(lstm_l1)
	return lstm_l1

callbacks = []

lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
callbacks.append(lr_callback)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)
callbacks.append(early_stopping_callback)

initializer = initializers.glorot_normal(seed=1)

tr_A = []
for i in range(number_of_tr_sample):
	X = trainInputs[i,:,:]
	y_ = trainTargets_[i,:]
	y = encode_onehot_label_setted(y_,nlabel+1)
	nAA = sum(y_ != nlabel)
	A = ds.generateAdjForMufoldNwsNconnect(ws,50,nAA)
	tr_A.append(A)
tr_A = np.array(tr_A)


te_A = []
for i in range(number_of_te_sample):
	X = testInputs[i,:,:]
	y_ = testTargets_[i,:]
	y = encode_onehot_label_setted(y_,nlabel+1)
	nAA = sum(y_ != nlabel)
	A = ds.generateAdjForMufoldNwsNconnect(ws,50,nAA)
	te_A.append(A)
te_A = np.array(te_A)

vl_A = []
for i in range(number_of_vl_sample):
	X = valInputs[i,:,:]
	y_ = valTargets_[i,:]
	y = encode_onehot_label_setted(y_,nlabel+1)
	nAA = sum(y_ != nlabel)
	A = ds.generateAdjForMufoldNwsNconnect(ws,50,nAA)
	vl_A.append(A)
vl_A = np.array(vl_A)

graph_conv_filters_tr = preprocess_adj_tensor_with_identity(tr_A, True)
graph_conv_filters_te = preprocess_adj_tensor_with_identity(te_A, True)
graph_conv_filters_vl = preprocess_adj_tensor_with_identity(vl_A, True)

n_units_conv_1 = Integer(low=20, high=130, name="n_unit_conv_1")
n_units_conv_2 = Integer(low=110, high=180, name="n_unit_conv_2")
n_units_gcn_1 = Integer(low=20, high=120, name="n_unit_gcn_1")
dr_rates = Categorical(categories=list([x / 10.0 for x in range(6)]),name="dr_rate")
n_units_dense_1 = Integer(low=100, high=500, name="n_unit_dense_1")
lrs = Real(low=1e-4, high=1e-1, prior='log-uniform',name="lr")
epochs = Integer(low=5, high=110, name="epoch")
#epochs = Integer(low=1, high=2, name="epoch")
batchs = Categorical(categories=list([2**x for x in range(0,4)]),name="batch")
n_units_lstm_1 = Integer(low=10, high=40, name="n_unit_lstm_1")

param_grid = [n_units_conv_1,n_units_conv_2,n_units_gcn_1,dr_rates,n_units_dense_1,lrs,epochs,batchs,n_units_lstm_1]

best_acc = 0.0

@use_named_args(dimensions=param_grid)
def my_model(n_unit_conv_1,n_unit_conv_2,n_unit_gcn_1,dr_rate,n_unit_dense_1,lr,epoch,batch,n_unit_lstm_1):

	X_input = Input(shape=(trainInputs.shape[1], trainInputs.shape[2]))
	graph_conv_filters_input = Input(shape=(graph_conv_filters_tr.shape[1], graph_conv_filters_tr.shape[2]))

	gcn_l_1_1 = GCNModule(X_input,n_unit_gcn_1,2,dr_rate,graph_conv_filters_input)
	cnn_l_1_1 = CNNModule(gcn_l_1_1,n_unit_conv_1,initializer)

	gcn_l_1_2 = GCNModule(cnn_l_1_1,n_unit_gcn_1,2,dr_rate,graph_conv_filters_input)
	cnn_l_1_2 = CNNModule(gcn_l_1_2,n_unit_conv_1,initializer)

	gcn_l_1_3 = GCNModule(cnn_l_1_2,n_unit_gcn_1,2,dr_rate,graph_conv_filters_input)
	cnn_l_1_3 = CNNModule(gcn_l_1_3,n_unit_conv_1,initializer)

	gcn_l_1_4 = GCNModule(cnn_l_1_3,n_unit_gcn_1,2,dr_rate,graph_conv_filters_input)
	cnn_l_1_4 = CNNModule(gcn_l_1_4,n_unit_conv_1,initializer)


	cnn_l_2_1 = CNNModule_2(cnn_l_1_4,n_unit_conv_2,initializer)
	cnn_l_2_1_end = concatenate([cnn_l_2_1,cnn_l_1_4])

	cnn_l_2_2 = CNNModule_2(cnn_l_2_1_end,n_unit_conv_2,initializer)
	cnn_l_2_2_end = concatenate([cnn_l_2_2,cnn_l_1_3])

	cnn_l_2_3 = CNNModule_2(cnn_l_2_2_end,n_unit_conv_2,initializer)
	cnn_l_2_3_end = concatenate([cnn_l_2_3,cnn_l_1_2])

	cnn_l_2_4 = CNNModule_2(cnn_l_2_3_end,n_unit_conv_2,initializer)
	cnn_l_2_4_end = concatenate([cnn_l_2_4,cnn_l_1_1])

	output = LSTMModule(cnn_l_2_4_end,n_unit_lstm_1,True,dr_rate)

	output = Dense(n_unit_dense_1,activation = acv_fnc)(output)

	output = Dense((nlabel+1))(output)
	output = Activation('softmax')(output)

	optimizer=Adam(lr=lr)
	model = Model(inputs=[X_input, graph_conv_filters_input], outputs=output)
	model.compile(loss = 'sparse_categorical_crossentropy',optimizer=optimizer)
	model.fit([trainInputs, graph_conv_filters_tr], trainTargets,validation_data=([valInputs,graph_conv_filters_vl],valTargets), callbacks=callbacks, batch_size=batch, epochs=epoch, verbose=2)

	p_label = model.predict([testInputs,graph_conv_filters_te],verbose=1)
	clean_targets,clean_label = pm.clean_prob(testTargets,p_label,nlabel)
	test_acc = pm.calc_acc_full_predict(testTargets,p_label,nlabel)

	global best_acc

	if test_acc > best_acc:
		print("Best Acc ",test_acc)
		best_acc = test_acc

		inf_file = open(dsName+".best_params","w")
		line = "acc\t" + str(test_acc)
		line = line + "\tlr\t" + str(lr) + "\tepoch\t" + str(epoch) + "\tdr_rate\t" + str(dr_rate) + "\tbatch\t" + str(batch)
		line = line + "\tn_unit_conv_1\t" + str(n_unit_conv_1) + "\tn_unit_conv_2\t" + str(n_unit_conv_2)
		line = line + "\tn_unit_gcn_1\t" + str(n_unit_gcn_1)
		line = line + "\tn_unit_dense_1\t" + str(n_unit_dense_1)
		line = line + "\tn_unit_lstm_1\t" + str(n_unit_lstm_1)
		
		inf_file.write(line)
		inf_file.close()

		np.savetxt(dsName + ".predict",clean_label)
		np.savetxt(dsName + ".targets",clean_targets)


	inf_file = open(dsName+".params","a")
	line = "acc\t" + str(test_acc)
	line = line + "\tlr\t" + str(lr) + "\tepoch\t" + str(epoch) + "\tdr_rate\t" + str(dr_rate) + "\tbatch\t" + str(batch)
	line = line + "\tn_unit_conv_1\t" + str(n_unit_conv_1) + "\tn_unit_conv_2\t" + str(n_unit_conv_2)
	line = line + "\tn_unit_gcn_1\t" + str(n_unit_gcn_1)
	line = line + "\tn_unit_dense_1\t" + str(n_unit_dense_1)
	line = line + "\tn_unit_lstm_1\t" + str(n_unit_lstm_1)

	inf_file.write(line + "\n")
	inf_file.close()

	return -test_acc

sr = gp_minimize(my_model,dimensions=param_grid,acq_func='EI',n_calls=50)
