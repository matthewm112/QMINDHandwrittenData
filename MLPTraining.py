import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense

(xtrain, ytrain),(xtest,ytest)=tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)
xtrain = xtrain/255

xtest = xtest/255
label_list= ["0","1","2","3","4","5","6","7","8","9",]
# fig = plt.figure()
# plt.imshow(xtrain[20], cmap='Greys')
# plt.title("label:{}".format(label_list[ytrain[20]]))
# plt.show()

model = models.Sequential()
# flattening out layers into single layer (single row array)
model.add(layers.Flatten(input_shape=(28, 28, 1)))
# normalizing data for training by setting min/max input to be considered
model.add(layers.Dense(254, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(xtrain, ytrain, epochs=50, batch_size=256, validation_data=(xtest, ytest))



# plotting the accuracy results from the training
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# predicting the data from the dataset
pred = model.predict(xtest)
# choosing the largest confidence value
pred = pred.argmax(axis=1)


# put data into confusion matrix function
disp = confusion_matrix(ytest, pred, labels=range(len(label_list)))
# setting figure size and colour
figure = plt.figure(figsize=(15, 12))
axis = figure.add_axes([.2, .2, .6, .6])
sns.heatmap(disp, annot=True, cmap='Blues', ax=axis)
# setting title and axes titles
axis.set_title('MLP Confusion Matrix')
axis.set_xlabel('Predicted Values')
axis.set_ylabel('Actual Values')
# ticket labels
axis.xaxis.set_ticklabels(label_list)
axis.yaxis.set_ticklabels(label_list)
plt.show()


# saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# #sample output
# probabilityModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probabilityModel.predict(xtest)
# print(predictions[35])
# print(np.argmax(predictions[35]))
# fig = plt.figure()
# plt.imshow(xtest[35], cmap='Greys')
# plt.title("label:{}".format(label_list[ytest[35]]))
# plt.show()
#
#
#
# cv_scores = []
# kFold = 10
# kVal = list(range(1, int(len(xtest))*((kFold-1)/kFold)))
# kVal= [k for k in kVal if k % 3 != 0]
#
# for k in kVal:
#     mlp = probabilityModel.predict(xtest)
#     scores = cross_val_score(mlp,xtest,ytest, cv=kFold, scoring='accuracy')
#     mean= scores.mean()
#     cv_scores.append(mean)
#     print(k, mean)
# mce = [1-xtest for xtest in cv_scores]
#
# print(f"Misclassification Error{mce} ")
# plt.plot(epochs, mce)
# plt.xlabel("Number of neighbours K")
# plt.ylabel("Misclassification Error")
# plt.show()