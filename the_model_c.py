import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Load CSV data
data = pd.read_csv("processed_tweets.csv")

# Split data into features (tweets) and labels (sentiment)
X = data['tweet']
y = data['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=1000)  # You can adjust the number of features

# Fit and transform on training data
X_train_vec = vectorizer.fit_transform(X_train)

# Convert the sparse matrix to a dense NumPy array
X_train_vec_dense = X_train_vec.toarray()

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Define CNN model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=X_train_vec_dense.shape[1]))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_vec_dense, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_vec, y_test, batch_size=64)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
