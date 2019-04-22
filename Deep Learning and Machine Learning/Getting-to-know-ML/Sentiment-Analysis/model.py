import keras 
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

model = Sequential()
model.add(Embedding(2500,128,input_length=X.shape[1],dropout=0.2))
model.add(LSTM(300, dropout_U=0.2,dropout_W=0.2))
model.add(Dense(2,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,verbose=2,batch_size=32)

print(model.evaluate(x_test,y_test)[1])
