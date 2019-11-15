import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score
import tensorflow as tf
from keras.layers import Input, Dense
from keras import Sequential


def linear_regression(x, y, n_splits=5, use_sklearn=True):
    kf = KFold(n_splits=n_splits, shuffle=True)
    mse = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lr = LinearRegression()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)


        #evaluate
        mse.append(np.mean((y_test - y_pred)**2))
      
    mse = np.sqrt(mse)
    return np.mean(mse), np.std(mse)

def softmax_regression(x, y, num_classes=3):
    y = np.array(y, dtype=int) - 1
    labels = np.zeros((len(y), len(np.unique(y))))
    print(np.unique(y))
    labels[np.arange(len(y)), y] = 1

    kf = KFold(n_splits=5, shuffle=True)
    losses, accs = [], []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        #fit 
        model = Sequential()
        model.add(Dense(len(np.unique(y)), activation='softmax',
                        input_shape=(x.shape[1],)))
        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=4, verbose=0)#, validation_split=0.2)

        #evaluate
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        losses.append(loss), accs.append(acc)

    return np.mean(acc), np.std(acc), np.mean(loss), np.std(loss)

def sigmoid_regression(x, y, num_classes=3):
    kf = KFold(n_splits=5, shuffle=True)
    losses = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #fit 
        model = Sequential()
        model.add(Dense(1, activation='sigmoid',
                        input_shape=(x.shape[1],)))
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=1, verbose=1)#, validation_split=0.2)

        #evaluate
        loss = model.evaluate(x_test, y_test, verbose=0)
        losses.append(loss)

    return np.mean(losses), np.std(losses)

def n_neighbours(x, y, n_neighbors=4):
    kf = KFold(n_splits=5, shuffle=True)
    mse = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #fit
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(x_train, y_train)
        #predict
        y_pred = model.predict(x_test)
        #evaluate
        mse.append(np.mean((y_test - y_pred)**2))
  
    mse = np.sqrt(mse)
    return np.mean(mse), np.std(mse)


def n_neighbours_classifier(x, y, n_neighbors=3, weights='uniform'):
    kf = KFold(n_splits=5, shuffle=True)
    acc = []
    recall = []
    prec = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #fit
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        model.fit(x_train, y_train)
        #evaluate
        y_pred = model.predict(x_test)
        acc.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        prec.append(precision_score(y_test, y_pred, average='weighted'))
  
    return np.mean(acc), np.std(acc), np.mean(recall), np.std(recall), np.mean(prec), np.std(prec)