import numpy as np
import h5py
import ast
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

class MyNeuralNetwork:
    def __init__(self, nn_architecture: dict = None):
        if nn_architecture is not None:
            self._layer_params = {}
            self._architecture = nn_architecture
            for idx, layer in enumerate(nn_architecture):
                layer_idx = idx + 1
                layer_input_size = layer['input_dim']
                layer_output_size = layer['output_dim']

                self._layer_params['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
                self._layer_params['b' + str(layer_idx)] = np.zeros((layer_output_size, 1))
        self._z_Outputs = {}
        self._a_Outputs = {}
        self._gradients = {}

    def back_prop(self, Y):
        self._gradients = {}
        a_curr = self._a_Outputs['a' + str(len(self._architecture))]
        m = a_curr.shape[1]
        error = -(1/m)*(Y - a_curr)

        for layer_idx_prev, layer in reversed(list(enumerate(self._architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer['activation']
            dA_curr = error
            if activ_function_curr == 'sigmoid':
                dZ_curr = self.sigmoid_derivate(a_curr)
            elif layer['activation'] == 'relu':
                dZ_curr = self.relu_derivate(a_curr)
            elif layer['activation'] == 'linear':
                dZ_curr = self.linear_derivate(a_curr)

            error = dA_curr * dZ_curr
            A_prev = self._a_Outputs['a' + str(layer_idx_prev)]
            dW_curr = np.dot(error, A_prev.T)
            db_curr = np.dot(error, np.ones((1, a_curr.shape[1])).T)
            self._gradients['W' + str(layer_idx_curr)] = dW_curr
            self._gradients['b' + str(layer_idx_curr)] = db_curr
            W_curr = self._layer_params['W' + str(layer_idx_curr)]

            error = np.dot(W_curr.T, error)
            a_curr = self._a_Outputs['a' + str(layer_idx_prev)]

    def update_parameters(self, l_rate=0.05):
        for key in self._gradients:
            self._layer_params[key] = self._layer_params[key] - l_rate*self._gradients[key]

    def feed_forward(self, X):
        self._z_Outputs = {}
        self._a_Outputs = {}
        a_prev = np.copy(X)
        if len(X.shape) < 2:
            a_prev = a_prev.reshape(a_prev.shape[0],1)

        self._a_Outputs['a0'] = X
        for idx, layer in enumerate(self._architecture):
            layer_number = idx + 1
            # z_curr = np.add(np.dot(self._layer_params[f'W{layer_number}'], a_prev), self._layer_params[f'b{layer_number}'])
            z_curr = np.dot(self._layer_params[f'W{layer_number}'], a_prev) + self._layer_params[f'b{layer_number}']
            self._z_Outputs[f'z{layer_number}'] = z_curr
            if layer['activation'] == 'sigmoid':
                a_curr = self.sigmoid(z_curr)
            elif layer['activation'] == 'relu':
                a_curr = self.relu(z_curr)
            elif layer['activation'] == 'linear':
                a_curr = np.array(z_curr, copy=True)
            self._a_Outputs[f'a{layer_number}'] = a_curr
            a_prev = a_curr

    def predict(self, X):
        self.feed_forward(X)
        return self.get_predicted()

    def get_predicted(self):
        return self._a_Outputs['a' + str(len(self._architecture))]

    def calculate_cost(self, Y, Y_hat):
        m = Y_hat.shape[1]
        return (1/(2*m)) * np.sum(np.square(Y - Y_hat))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivate(self, A):
        return A * (1 - A)

    def relu(self, Z):
        return Z * (Z > 0)

    def relu_derivate(self, A):
        res = np.array(A, copy=True)
        res[res <= 0] = 0
        res[res > 0] = 1
        return res

    def linear_derivate(self, A):
        res = np.array(A, copy=True)
        res[True] = 1
        return res

    def train(self, X, Y, epochs, l_rate=0.001, batch_size=1):
        if len(X.shape) < 2:
            number_of_samples = 1
        else:
            number_of_samples = X.shape[1]
        iterations = number_of_samples // batch_size
        for epoch in range(epochs):
            for i in range(iterations):
                x_batch = X[:,i*batch_size:(i+1)*batch_size]
                y_batch = Y[i*batch_size:(i+1)*batch_size]

                self.feed_forward(x_batch)
                print(f'Epoch {epoch} cost: {self.calculate_cost(y_batch, self.get_predicted())}')
                self.back_prop(y_batch)
                self.update_parameters(l_rate)

    def load_from_file(self, file_name: str = 'model.hdf5'):
        self._architecture = []
        self._layer_params = {}
        with h5py.File(file_name, "r") as f:
            model_info = ast.literal_eval(f.get('model_info')[...].tolist())
            number_of_layers = model_info['number_of_layers']
            for i in range(number_of_layers):
                layer_idx = i+1
                self._architecture.append(ast.literal_eval(f.get(f'/layers/layer{layer_idx}')[...].tolist()))
                self._layer_params[f'W{layer_idx}'] = f.get(f'/layers/weights/W{layer_idx}')[()]
                self._layer_params[f'b{layer_idx}'] = f.get(f'/layers/biases/b{layer_idx}')[()]

            print('complete')

    def save_to_file(self, file_name: str = 'model.hdf5'):
        number_of_layers = len(self._architecture)
        with h5py.File(file_name, "w") as f:
            f.create_dataset('model_info', data=str({'number_of_layers': number_of_layers}))
            for i in range(number_of_layers):
                layer_idx = i + 1
                f.create_dataset(f'/layers/layer{layer_idx}', data=str(self._architecture[i]))

                f.create_dataset(f'/layers/weights/W{layer_idx}', data=self._layer_params[f'W{layer_idx}'])

                f.create_dataset(f'/layers/biases/b{layer_idx}',
                                 data=self._layer_params[f'b{layer_idx}'])



#
# nn = MyNeuralNetwork([{'input_dim': 2, 'output_dim': 3, 'activation': 'sigmoid'},
#                       {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}])
# nn = MyNeuralNetwork()
# nn.load_from_file("mytestfile.hdf5")
# nn.load_from_file()
# nn.save_to_file()
# nn.load_from_file()
# nn.save_to_file()
# nn.load_from_file()
# nn.train(np.array([[0, 0, 1, 1],
#                    [0, 1, 0, 1]]), np.array([[0, 1, 1, 0]]), 100000, 5)
#
# print(nn.predict(np.array([[0, 0, 1, 1],                   [0, 1, 0, 1]])))
# print(np.array([[1],[0]]).shape)
# nn.save_to_file()
df = pd.read_csv('./datasets/housepricedata.csv')
dataset = df.values
X = dataset[:, 0:10]
Y = dataset[:, 10]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

# with open('test.npy', 'wb') as f:
#     np.save(f, X_test)
#     np.save(f, Y_test)
# print(X_train[0].shape)
# with open('test.npy', 'rb') as f:
#     X_test = np.load(f)
#     Y_test = np.load(f)
# othr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# sk = np.ones(10)
# sk = sk.reshape((sk.shape[0], 1))
# print(np.dot(othr, sk))

# print(np.transpose(X_train[0:2]))

# nn = MyNeuralNetwork([{'input_dim': 10, 'output_dim': 32, 'activation': 'sigmoid'},
# {'input_dim': 32, 'output_dim': 32, 'activation': 'sigmoid'},
# {'input_dim': 32, 'output_dim': 1, 'activation': 'sigmoid'}])
# # nn = MyNeuralNetwork()
# # nn.load_from_file()
# # print(nn.predict(X_test[0].reshape((X_test[1].shape[0],1))))
# # nn.predict()
# nn.train(X_train.T, Y_train, 5000, 0.005, 1)
# nn.save_to_file()
# # print(nn.predict(X_test[1:10].T), Y_test[1:10])
# print(nn.predict(X_test[1:10].T), Y_test[1:10])
# print(X_test[1:10].T.shape[0])
# print(nn.predict(X_test[3]), Y_test[3])

# test = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# batch_size = 1
# iterations = test.size // batch_size
# print(iterations)
# for i in range(iterations):
#     print(test[i*batch_size:(i+1)*batch_size])


# x1 = ones[]
# x2 = np.arange(3.0)
# print(x1)
# print(x2)
# print(x1 + x2)
# nn.save_to_file()


# data = pd.read_csv('./datasets/iris.csv')
#
# data.loc[data["Species"]=="Iris-setosa","Species"]=0
# data.loc[data["Species"]=="Iris-versicolor","Species"]=1
# data.loc[data["Species"]=="Iris-virginica","Species"]=2
#
# print(data.head())
#
# data=data.iloc[np.random.permutation(len(data))]
# print(data.head())
#
# X=data.iloc[:,1:5].values
# y=data.iloc[:,5].values
# print("Shape of X",X.shape)
# print("Shape of y",y.shape)
# print("Examples of X\n",X[:3])
# print("Examples of y\n",y[:3])
#
# X_normalized=normalize(X,axis=0)
#
# print("Examples of X_normalised\n",X_normalized[:3])
#
# total_length=len(data)
# train_length=int(0.8*total_length)
# test_length=int(0.2*total_length)
#
# X_train=X_normalized[:train_length]
# X_test=X_normalized[train_length:]
# y_train=y[:train_length]
# y_test=y[train_length:]
#
# print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
# print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])

train = pd.read_csv('./datasets/train_clean.csv')
test = pd.read_csv('./datasets/test_clean.csv')
df = pd.concat([train, test], axis=0, sort=True)


df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes
categorical = ['Embarked', 'Title']
for var in categorical:
    df = pd.concat([df,
                    pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
print(df.head())

continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
scaler = StandardScaler()

for var in continuous:
    df[var] = df[var].astype('float64')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

print(df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1))
X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1).to_numpy()
y_train = df[pd.notnull(df['Survived'])]['Survived'].to_numpy()
X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1).to_numpy()


nn = MyNeuralNetwork([{'input_dim': X_train.shape[1], 'output_dim': 8, 'activation': 'linear'},
{'input_dim': 8, 'output_dim': 1, 'activation': 'linear'},
{'input_dim': 1, 'output_dim': 1, 'activation': 'sigmoid'}])

nn.train(X_train.T, y_train, 5000, 0.025, 32)

print(nn.predict(X_train[0:10].T), y_train[0:10])
