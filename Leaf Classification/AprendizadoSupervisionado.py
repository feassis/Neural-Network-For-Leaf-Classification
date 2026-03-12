import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def squared_error(y_pred, y_true):
    return (y_pred - y_true) ** 2

def one_hot(y, num_classes):
    y = y.to_numpy().astype(int) - 1
    
    one_hot_matrix = np.zeros((len(y), num_classes))
    
    one_hot_matrix[np.arange(len(y)), y] = 1
    
    return one_hot_matrix

num_inputs = 14
num_output_types = 36
num_neurons = 36
learning_rate = 0.01
epoch = 5000

df = pd.read_csv("leaf.csv", header=None)
df = df.drop(columns=[1])

df.columns = range(df.shape[1])

atributos = df.iloc[:, 1:]

#normalizacao
df.iloc[:, 1:] = (atributos - atributos.min()) / (atributos.max() - atributos.min())

df = df.sample(frac=1).reset_index(drop=True)

#df = df.sample(frac=1, random_state=42).reset_index(drop=True) # caso queri afixar a aleatoriedade

tabelaParaTestesReal = df.groupby(0).head(3)
tabelaParaTreinamentoReal = df.drop(tabelaParaTestesReal.index)

Y_Test = tabelaParaTestesReal.iloc[:, 0]
Y_Test_Oh = one_hot(Y_Test, 36)

X_Test = tabelaParaTestesReal.iloc[:, 1:]
X_Test[num_inputs+1] = 1

Y_Train = tabelaParaTreinamentoReal.iloc[:, 0]
Y_Train_Oh = one_hot(Y_Train, 36)

X_Train = tabelaParaTreinamentoReal.iloc[:, 1:]
X_Train[num_inputs+1] = 1

# +1 por causa do bias
W = np.random.rand(num_neurons, num_inputs + 1) - 0.5

for k in range(epoch):
    print("epoca ", k)
    for i in range(len(X_Train)):
        x = X_Train.iloc[i]
        
        #multiplicação com W transposta
        z = np.dot(x, W.T)
        
        ativacao = sigmoid(z)
        
        diferenca = ativacao - Y_Train_Oh[i]
        erroDerivado = diferenca * ativacao * (1 - ativacao)
        
        deltaW = - learning_rate * 2 * np.outer(erroDerivado, x)
        
        W += deltaW
            
predictions = []

for i in range(len(X_Test)):
    x = X_Test.iloc[i]
    
    z = np.dot(x, W.T)

    pred = sigmoid(z)
    
    predictions.append(pred)
    
predictions = np.array(predictions)

indices = np.argmax(predictions, axis=1)

one_hot_pred = np.zeros_like(predictions)

one_hot_pred[np.arange(len(predictions)), indices] = 1

conf_matrix = np.zeros((36,36), dtype=int)

acerto = 0

for i in range(len(one_hot_pred)):
    print("Teste")
    real = np.argmax(Y_Test_Oh[i])
    print(real)
    pred = np.argmax(one_hot_pred[i])
    print(pred)
    
    if(real == pred):
        acerto += 1

    conf_matrix[real, pred] += 1

porcentagem = (acerto / len(X_Test)) * 100
print(f"Acertos: {acerto}/{len(X_Test)}")
print(f"Porcentagem: {porcentagem:.2f}%")