import net
import data
import metric
import numpy
import keras

numpy.random.seed(0)

# data
train_users, train_x, test_users, test_x = data.load_data()
#train_users_1 = train_users
#train_users_1[train_users_1 != 0] = 0
#train_x_1 = train_x
#train_x_1[train_x_1 != 0] = 0

train_x_users = numpy.array(train_users, dtype=numpy.int32).reshape(len(train_users), 1)
#train_x_users_1 = numpy.array(train_users_1, dtype=numpy.int32).reshape(len(train_users_1), 1)

test_x_users = numpy.array(test_users, dtype=numpy.int32).reshape(len(test_users), 1)

# model
model = net.create(I=train_x.shape[1], U=len(train_users) + 1, K=50,
                   hidden_activation='relu', output_activation='sigmoid', q=0.80, l=0.01)
#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, train_x_users], y=train_x,
                    batch_size=128, nb_epoch=1000, verbose=1,
                    validation_data=[[test_x, test_x_users], test_x])

# predict
pred = model.predict(x=[train_x, train_x_users])
# remove watched items from predictions
pred *= (train_x == 0)
pred_arg = numpy.argsort(pred)

N = [1, 5, 10]
for i in range(len(N)):
    sr = 0.000
    for j in range(pred.shape[0]):
        sr += metric.apk(test_x[j], pred_arg[j, -N[i]:], N[i])
    print("mAP at Top@{:d} Recommendation is: {:f}".format(N[i], sr / pred.shape[0]))
