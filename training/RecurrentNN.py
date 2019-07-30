from training import datahelper
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, GRU, CuDNNGRU, CuDNNLSTM

look_back = 2

# x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)
# with open('df_backup_full.s', 'wb') as f:
#     pickle.dump([x, y], f)

with open('df_backup_full.s', 'rb') as f:
    x, y = pickle.load(f)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = np.reshape(x_train.values, (x_train.shape[0], 9, 1))
x_test = np.reshape(x_test.values, (x_test.shape[0], 9, 1))

model = Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, input_shape=(9, 1)))
model.add(CuDNNGRU(100))
model.add(Dense(4))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

model.fit(x_train, y_train, epochs=500, batch_size=32)

# filename = '../models/recurrent_nn_model.sav'
# print('Saving model to file ', filename)
# with open(filename, 'wb') as h:
#     pickle.dump(model, h)

y_pred = model.predict(x_test)


print('R^2 = ' + str(r2_score(y_test, y_pred)))
print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print()

m = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(y.columns.values), m):
    print(col, ": ", err)