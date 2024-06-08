# Importando bibliotecas
from keras import layers, models
import numpy as np

class CNN:
    # Método estático, retorna uma class CNN instânciada
    @staticmethod
    def CriarRedeNeural():
        return CNN()
    
    def __init__(self):
        self.model = models.Sequential()
        # Ajuste do input_shape para 12x10x1 (altura, largura, canais)
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 10, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        # Saída com 26 classes para as letras
        self.model.add(layers.Dense(26, activation='softmax'))
    
    # Função para compilação da rede: determina algoritmo otimizador, perda e métrica
    def compilar(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
    # Função que inicia treinamento da rede
    def iniciarTreinamento(self, dados_treinamento, rotulos_treinamento, epocas, tamanho_lote, dados_validacao):
        return self.model.fit(x = dados_treinamento,
                                 y = rotulos_treinamento, 
                                 epochs = epocas, 
                                 batch_size = tamanho_lote, 
                                 validation_data = dados_validacao)

# Carrega os dados e rotulos
dados = np.load('dados/X.npy')
rotulos = np.load('dados/Y_classe.npy')

# Preparar os dados de treinamento, validação e teste
dados_treinamento = np.array(dados[:900])
rotulos_treinamento =  np.array(rotulos[:900])
dados_validacao =  np.array(dados[900:1150])
rotulos_validacao =  np.array(rotulos[900:1150])
dados_teste =  np.array(dados[1150:])
rotulos_teste =  np.array(rotulos [1150:])


# Compilar e treinar a rede neural 
cnn = CNN.CriarRedeNeural()
cnn.compilar()

# Inicia treinamento da rede
historia = cnn.iniciarTreinamento(
    dados_treinamento = dados_treinamento,
    rotulos_treinamento = rotulos_treinamento,
    epocas = 50,
    tamanho_lote = 300,
    dados_validacao = (dados_validacao, rotulos_validacao)
)

# Pega métricas da rede treinada
teste_perda, teste_acuracia = cnn.model.evaluate(dados_teste, rotulos_teste, verbose = 2)
print(f'\n Acurácia de teste: {teste_acuracia}; Perda: {teste_perda}')
