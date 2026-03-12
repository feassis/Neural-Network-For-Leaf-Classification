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

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def one_hot(y, num_classes):
    
    y = y.to_numpy().astype(int) - 1
    
    one_hot_matrix = np.zeros((len(y), num_classes))
    
    one_hot_matrix[np.arange(len(y)), y] = 1
    
    return one_hot_matrix

num_inputs = 14
num_output_types = 36
num_neurons = 36
learning_rate = 0.045
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


erros = []

for k in range(epoch):
    print("epoca ", k)
    erro_epoca = 0

    for i in range(len(X_Train)):
        x = X_Train.iloc[i]
        
        #multiplicação com W transposta
        z = np.dot(x, W.T)
        
        ativacao = softmax(z)
        
        diferenca = ativacao - Y_Train_Oh[i]
        
        mse = np.mean(diferenca**2)
        
        erro_epoca += mse
        
        deltaW = - learning_rate * np.outer(diferenca, x)
        
        W += deltaW
        
        
    erro_epoca = erro_epoca / len(X_Train)
    
    erros.append(erro_epoca)
    
    print(f"epoca {k} erro {erro_epoca}")

plt.plot(erros)
plt.xlabel("Epoca")
plt.ylabel("Erro (MSE)")
plt.title("Evolucao do erro durante o treinamento")

plt.show()

            
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
    real = np.argmax(Y_Test_Oh[i])
    pred = np.argmax(one_hot_pred[i])
    
    if(real == pred):
        acerto += 1

    conf_matrix[real, pred] += 1
    
porcentagem = (acerto / len(X_Test)) * 100
print(f"Acuracia: {acerto}/{len(X_Test)}")
print(f"Porcentagem: {porcentagem:.2f}%")

num_classes = conf_matrix.shape[0]

for k in range(num_classes):

    TP = conf_matrix[k,k]

    FN = np.sum(conf_matrix[k,:]) - TP

    FP = np.sum(conf_matrix[:,k]) - TP

    TN = np.sum(conf_matrix) - TP - FN - FP
    
    precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    recall = TP / (TP + FN) if (TP+FN) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0

    print(f"Classe {k}")
    print("TP:", TP)
    print("FN:", FN)
    print("FP:", FP)
    print("TN:", TN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    
    print()
    
    
