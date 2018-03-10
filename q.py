import CDAE_model_establishment
import input_dataset_movie_lens
from sklearn.metrics import average_precision_score

import numpy as np

np.random.seed(0)
# data
train_users, train_x, test_users, test_x = input_dataset_movie_lens.load_data()
train_x_users = np.array(train_users, dtype=np.int32).reshape(len(train_users), 1)
test_x_users = np.array(test_users, dtype=np.int32).reshape(len(test_users), 1)

# model
model = CDAE_model_establishment.create(I=train_x.shape[1], U=len(train_users) + 1, K=50,
                                        hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, train_x_users], y=train_x,
                    batch_size=128, nb_epoch=1000, verbose=1,
                    validation_data=[[test_x, test_x_users], test_x])

# predict
pred = model.predict(x=[train_x, np.array(train_users, dtype=np.int32).reshape(len(train_users), 1)])
pred = pred * (train_x == 0)  # remove watched items from predictions
pred_arg = np.argsort(pred)


print(type(pred))
print(type(pred_arg))
print(pred_arg[:, -1:])
