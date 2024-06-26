# Importando bibliotecas
import json
import os
from keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

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
        self.model.add(layers.MaxPooling2D((2, 2), 2))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2), 2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(26, activation='softmax'))
    
    # salvarPesos salva os pesos da rede. 
    def salvarPesos(self, nome_arquivo, pesos):
        for i, peso in enumerate(pesos):
            np.save(f'informacoes_modelo/pesos/{nome_arquivo}_camada_{i}.npy', peso) 
            
    # salvarArquiteturaModelo salva a arquitetura do modelo
    def salvarArquiteturaModelo(self):
        modelo_formato_json = self.model.to_json()
        with open('informacoes_modelo/arquitetura/arquitetura_do_modelo.json', 'w') as json_file:
            json_file.write(modelo_formato_json)
    
    # salvarErroIteracaoTreinamento
    def salvarErroIteracaoTreinamento(self, historia):
        # Salvar o histórico de treinamento
        dataframe = pd.DataFrame(historia.history)
        dataframe.to_csv('informacoes_modelo/erro_durante_treinamento/historia_treinamento.csv', index=False)
    
    # salvarHiperparametros salva os hiperparametros da rede.
    def salvarHiperparametros(self):
        # Recuperando hiperparâmetros paras salvar a configuração da rede.
        hiperparametros = {
            'optimizer': self.model.optimizer.get_config(),
            'loss': self.model.loss,
            'metrics': self.model.metrics_names,
            'layers': [layer.get_config() for layer in self.model.layers]
        }
        
        # Define a pasta onde os hiperparâmetros serão salvos
        pasta = 'informacoes_modelo/hiperparametros/'
        os.makedirs(pasta, exist_ok=True)

        # Caminho do arquivo onde os hiperparâmetros serão salvos
        caminho_do_arquivo = os.path.join(pasta, 'hiperparametros.json')

        # Salva os hiperparâmetros no arquivo JSON
        with open(caminho_do_arquivo, 'w') as f:
            json.dump(hiperparametros, f, indent=4)

        print(f"Hiperparâmetros salvos em {caminho_do_arquivo}")

    # gerarMatrizConfusao recebe os dados de teste e gera, plota e salva a matriz de confusao para os dados.
    def gerarMatrizConfusao(self, dados_teste, rotulos_teste):
        # Gera predicoes para os dados de teste
        predicoes = cnn.model.predict(dados_teste)
        
        # Salva as predicoes da rede para os dados de teste.
        np.save('informacoes_modelo/saidas_dados_teste/predicoes_teste.npy', predicoes)
        
        # Itera sobre as predicoes e pega a letra rotulada para o respectivo dado pela rede para todos os dados.
        letra_rotulo_predicao = []
        for predicao in predicoes:
            letra_rotulo_predicao.append(np.argmax(predicao))

        # Itera sobre os rotulos e pega cada letra esperada que a rede rotule o dado para todos os dados.
        letra_rotulo_teste = []
        for rotulo in rotulos_teste:
            letra_rotulo_teste.append(np.argmax(rotulo))

        # Com as letras rotulada pela rede e as letras esperadas é gerada a matriz de confusão
        matriz = confusion_matrix(letra_rotulo_teste, letra_rotulo_predicao)

        # Gera de letras que representam nossos rotulos
        rotulos = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        # Cria uma dataframe com pandas da matriz de confusão
        dataframe = pd.DataFrame(matriz, index=rotulos, columns=rotulos)

        # Salva em excel a matriz de confusão na pasta: matriz_de_confusao
        dataframe.to_excel('matriz_de_confusao/matriz_excel.xlsx', sheet_name='Matriz de Confusão')
        print('\nMatriz salva em matriz_de_confusao/matriz_excel.xlsx\n')
        
        # Configura a dimensão, letras nas colunas e linhas, cores e números inteiros para matriz.
        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz, annot=True, fmt='', cmap='viridis', xticklabels=rotulos, yticklabels=rotulos)

        # Ajusta legendas.
        plt.xlabel('Colunas')
        plt.ylabel('Linhas')
        plt.title('Matriz de confusão 26x26 com rótulos de A-Z')

        # Plota matriz
        plt.show()
    
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
dados_validacao =  np.array(dados[800:1050])
rotulos_validacao =  np.array(rotulos[800:1050])
dados_teste =  np.array(dados[1050:])
rotulos_teste =  np.array(rotulos [1050:])

# Criação da rede e compilação. 
cnn = CNN.CriarRedeNeural()
cnn.compilar()

# Salva pesos iniciais e arquitetura da rede
cnn.salvarPesos('pesos_iniciais', cnn.model.get_weights())
cnn.salvarArquiteturaModelo()

# Inicia treinamento da rede e recebe informações referentes ao erro, acurácia, camadas na variável historia.
historia = cnn.iniciarTreinamento(
    dados_treinamento = dados_treinamento,
    rotulos_treinamento = rotulos_treinamento,
    epocas = 50,
    tamanho_lote = 150,
    dados_validacao = (dados_validacao, rotulos_validacao)
)

# Cálcula o erro e acurácia do modelo treinado utilizando o conjuto de teste.
erro, acuracia = cnn.model.evaluate(dados_teste, rotulos_teste)

print(f'\nMétricas calculadas no conjunto de teste \nAcurácia: {acuracia} - Erro quadrático médio: {erro}\n')

# Gera a matriz de confusao salvando em excel e plotando para visualização
cnn.gerarMatrizConfusao(dados_teste, rotulos_teste)

# Salva as informações relevantes do modelo.
cnn.salvarErroIteracaoTreinamento(historia)
cnn.salvarPesos('pesos_finais', cnn.model.get_weights())
cnn.salvarHiperparametros()