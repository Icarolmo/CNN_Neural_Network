# Importando bibliotecas
import json
import os
from keras import layers, models
import numpy as np

class CNN:
    # Método estático, retorna uma class CNN instânciada
    @staticmethod
    def CriarRedeNeural():
        return CNN()
    
    def __init__(self):
        self.model = models.Sequential()
        # Descrição da arquitetura e organização das camadas da rede:
        # 1ª - Conv2D (Convolucional Bidimensional): aplica 32 filtros (kernels) de tamanho 3x3 com função de ativação relu.
        # 2ª - MaxPooling2D: aplica operação de pooling utilizando filtros 2x2.
        # 3ª - Conv2D (Convolucional Bidimensional): aplica 64 filtros (kernels) de tamanho 3x3 com função de ativação relu.
        # 4ª - MaxPooling2D: aplica operação de pooling utilizando filtros 2x2.
        # 5ª - Flatten: transforma o vetor com dados com mais de uma dimensão em um vetor unidimensional mantendo as caracteristicas mais importantes.
        # 6ª - Dense (Camada densa): Camada totalmente conectada de 64 neurônios com função de ativação relu.
        # 7ª - Dense (Camada densa): Última camada: Saída com 26 neurônios, cada um representando uma classe (letra).
        # É utilizada a função softmax para rotular o dado com uma classe.
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 10, 1))) # Ajuste do input_shape para 12x10x1 (Algura, Lagura e Canais de cores).
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(26, activation='softmax'))
    
    # Função para salvamento dos hiperparametros da rede.
    def salvarHiperparametros(self):
        # Recuperando hiperparâmetros paras salvar a configuração da rede.
        hiperparametros = {
            'optimizer': self.model.optimizer.get_config(),
            'loss': self.model.loss,
            'metrics': self.model.metrics_names,
            'layers': [layer.get_config() for layer in self.model.layers]
        }
        
        # Define a pasta onde os hiperparâmetros serão salvos
        pasta = 'hiperparametros'
        os.makedirs(pasta, exist_ok=True)

        # Caminho do arquivo onde os hiperparâmetros serão salvos
        caminho_do_arquivo = os.path.join(pasta, 'hiperparametros.json')

        # Salva os hiperparâmetros no arquivo JSON
        with open(caminho_do_arquivo, 'w') as f:
            json.dump(hiperparametros, f, indent=4)

        print(f"Hiperparâmetros salvos em {caminho_do_arquivo}")

    
    # Função para compilação da rede: determina algoritmo otimizador, função para cálculo do erro e métrica utilizada.
    def compilar(self):
        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])
        
    # Função que inicia treinamento da rede chamando método fit de treinamento do Keras.
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
dados_treinamento = np.array(dados[:800])
rotulos_treinamento =  np.array(rotulos[:800])
dados_validacao =  np.array(dados[800:1150])
rotulos_validacao =  np.array(rotulos[800:1150])
dados_teste =  np.array(dados[1150:])
rotulos_teste =  np.array(rotulos [1150:])

# Criação da rede e compilação. 
cnn = CNN.CriarRedeNeural()
cnn.compilar()

# Inicia treinamento da rede e recebe informações referentes ao erro, acurácia, camadas na variável historia.
historia = cnn.iniciarTreinamento(
    dados_treinamento = dados_treinamento,
    rotulos_treinamento = rotulos_treinamento,
    epocas = 50,
    tamanho_lote = 500,
    dados_validacao = (dados_validacao, rotulos_validacao)
)

# Cálcula o erro e acurácia do modelo treinado utilizando o conjuto de teste.
erro, acuracia = cnn.model.evaluate(dados_teste, rotulos_teste)

print(f'\nMétricas calculadas no conjunto de teste \nAcurácia: {acuracia} - Erro quadrático médio: {erro}\n')

# Salva os hiperparametros e configuração da rede.
cnn.salvarHiperparametros()
