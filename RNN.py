import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Step 2: Import Data


# Generating Sample Dataset
r_file = '10minBUstats.csv'

N = len(pd.read_csv(r_file))
Tp = int(.15 * N)

t = np.arange(0, N)
x = training_data = pd.read_csv(r_file)
df = pd.DataFrame(x)

df.head()
# Quick Plot
plt.plot(df)
plt.show()

# Step 3: Format Data

# Splitting into testing and training

values=df.values
train, test = values[0:Tp,:], values [Tp:N,:]
# The second colen is the step

# 0 = start. Tp = 800. For training N = 1000
# Is to represent step
# Reshaping
step = 120

# Add the step into the testing and training sets
test = np.append(test,np.repeat(test[-1,],step))
train = np.append(test,np.repeat(test[-1,],step))

# Convert into a matrix from a timeseries
def convertToMatrix(data, step):
    X, Y =[], []
    for i in range(len(data)-step):
        d=i+step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

# Reshape trainX and testX to fit with the Keras model. RNN model requires 3-D input data
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape
testX.shape
# Create the model
model = Sequential()
model.add(LSTM(units=64, input_shape=(1, step), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=2)
trainScore = model.evaluate(trainX, trainY, verbose=0)
print("Train Score: %.2f MSE (%.2f RMSE)" % (trainScore, np.sqrt(trainScore)))
# Predict the test data
testPredict = model.predict(testX)
# Plot the results
plt.plot(testY,color='red', label = 'Real Message #')
plt.plot(testPredict,color='blue', label = 'Predicted Message #')
plt.title("Message # Prediction")
plt.xlabel("Time")
plt.ylabel("Messages Recieved")
plt.legend()
plt.show()

# Define predicted
predicted = testPredict