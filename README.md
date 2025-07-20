# Análise Comparativa de Verificação de Locutor com k-NN e SVM

Este repositório contém o projeto desenvolvido para a disciplina de Introdução à Inteligência Artificial, do curso de Bacharelado em Tecnologia da Informação do Instituto Metrópole Digital (IMD/UFRN).

**Aluna:** Rosângela D'Avilla

## 🎯 Objetivos e Aplicações

O objetivo central deste projeto é desenvolver e avaliar um sistema de verificação de locutor. Em vez de identificar quem está falando (identificação de locutor), o sistema foca em responder à seguinte pergunta:

> "Estes dois áudios pertencem à mesma pessoa?" (Sim/Não)

Para isso, o projeto realiza uma análise comparativa entre dois algoritmos clássicos de aprendizado de máquina, **k-Nearest Neighbors (k-NN)** e **Support Vector Machine (SVM)**, para determinar qual deles oferece o melhor desempenho na tarefa proposta.

A principal aplicação de um sistema como este está em áreas **forense e criminal**, onde acontece muitas vezes de precisar saber se a voz que aparece em um áudio ou vídeo é de determinado acusado/criminoso.

## 📚 Dataset Utilizado: Common Voice (Mozilla)

Para treinar e testar os modelos, foi utilizado o dataset **Common Voice Corpus 22.0**, mantido pela Mozilla. Este é um dataset público e colaborativo, com as seguintes características:

- **Licença:** CC-0 (Domínio Público)  
- **Formato:** Arquivos de áudio em `.mp3` acompanhados de metadados em um arquivo `.tsv`  
- **Volume de Dados:** A versão utilizada conta com **185 horas de áudio validadas**, de **3.759 locutores diferentes**

O dataset apresenta uma diversidade de locutores, embora com um desbalanceamento demográfico notável, com predominância de **locutores masculinos (68%)** e da **faixa etária de 20 a 29 anos (36%)**.

## 🤖 Arquitetura do Agente e Modelagem

### Representação de Conhecimento

O principal desafio é transformar um sinal de áudio em dados que um modelo de IA possa entender. Para isso, utilizamos a técnica de **Coeficientes Cepstrais de Frequência Mel (MFCC)**. O MFCC processa o áudio e o converte em uma matriz numérica que funciona como uma "assinatura vocal".

O conhecimento que o agente utiliza para a decisão não são os MFCCs brutos, mas sim a **diferença absoluta** calculada entre as matrizes MFCC de um par de áudios. Essa diferença é o vetor de **features** de entrada para os modelos.

### Algoritmos de IA

Foram implementados e comparados dois agentes baseados em algoritmos de aprendizado supervisionado:

- **k-Nearest Neighbors (k-NN):** Um modelo baseado em instância que classifica um novo dado com base na classe de seus "vizinhos" mais próximos.  
  - *Implementação:* `KNeighborsClassifier` da biblioteca **scikit-learn**  
  - *Parâmetro Principal:* `n_neighbors = 5`

- **Support Vector Machine (SVM):** Um modelo que busca encontrar o melhor hiperplano para separar os dados em diferentes classes.  
  - *Implementação:* `SVC` da biblioteca **scikit-learn**  
  - *Parâmetro Principal:* `kernel = 'rbf'` (função de base radial)

### Modelagem PEAS

A tarefa do agente foi modelada utilizando o framework **PEAS** (Performance, Environment, Actuators, Sensors):

- **P (Performance / Desempenho):** Acurácia do sistema em classificar corretamente se duas amostras de áudio são do mesmo locutor ("Sim" / "Não"). A **Matriz de Confusão** é usada para uma análise detalhada dos acertos e erros.  
- **E (Environment / Ambiente):** Pares de áudios reais do dataset Common Voice, que possuem variações de qualidade, ruído de fundo e características de diferentes locutores.  
- **A (Actuators / Atuadores):** O sistema processa um par de áudios e produz uma saída binária: `1` se pertencerem ao mesmo locutor ou `0` caso contrário.  
- **S (Sensors / Sensores):** O agente "percebe" o ambiente através de arquivos de áudio brutos (em formato `.mp3`) e, a partir deles, realiza a extração dos coeficientes MFCC.

## 🛠️ Pipeline do Projeto

O processo de construção do modelo seguiu 4 etapas principais:

1. **Seleção de Locutores Válidos:**  
   Foram filtrados apenas os locutores que possuíam no mínimo 4 clipes de áudio no dataset. Isso resultou em **93 locutores válidos**.

2. **Criação dos Pares de Áudios:**  
   - **Pares Positivos (Label = 1):** Dois áudios distintos do mesmo locutor.  
   - **Pares Negativos (Label = 0):** Um áudio de um locutor e outro de um locutor diferente, escolhido aleatoriamente.

3. **Pré-processamento e Extração de Features:**  
   Para cada par, os **MFCCs** foram extraídos e a **diferença absoluta** entre eles foi calculada, resultando em um vetor de entrada.

4. **Treinamento e Teste:**  
   O dataset final foi dividido em **80% para treinamento (1004 amostras)** e **20% para teste (252 amostras)**.

## 📊 Métricas e Desempenho

A performance dos modelos foi avaliada com base na **Acurácia** e na análise da **Matriz de Confusão**.

### Resultados

| Modelo | Acurácia | VN  | FP  | FN  | VP  |
|--------|----------|-----|-----|-----|-----|
| k-NN   | 82.9%    | 157 | 30  | 13  | 52  |
| SVM    | 88.5%    | 176 | 11  | 18  | 47  |

O modelo **SVM** alcançou uma acurácia superior (**88.5%**), demonstrando ser mais eficaz para esta tarefa. Ele também teve **menos Falsos Positivos** (11 contra 30 do k-NN), ou seja, errou menos ao dizer que dois locutores diferentes eram a mesma pessoa.

## ⚠️ Limitações e Dificuldades

### Limitações do Projeto e da Arquitetura

- **Representação de Features Simplificada:** A diferença absoluta entre MFCCs é uma abordagem simplificada que perde informações temporais.
- **Sensibilidade dos Algoritmos:**  
  - O k-NN depende fortemente da escolha de `k`.  
  - O SVM depende do `kernel` e dos hiperparâmetros.  
- **Abordagem Clássica vs. Moderna:** Técnicas mais recentes como **CNNs** em espectrogramas ou **Transformers** podem gerar melhores resultados.

### Dificuldades no Processo

- **Dataset Desbalanceado: A predominância de locutores masculinos e jovens pode ter criado um viés no modelo.
- **Features Simplificadas: A "diferença absoluta" dos MFCCs perde detalhes temporais da voz.
- **Abordagem Clássica: Modelos modernos de Deep Learning costumam ter performance superior.

## ✨ Sugestões de Melhorias

- **Usar PyCaret: Se eu tivesse usado o PyCaret teria comparado mais modelos.
- **Balancear melhor os dados.
- **Talvez tentar tratar os áudios antes (Redução de ruído, etc...)




Explorar Modelos de Deep Learning: Implementar uma Rede Neural Convolucional (CNN) que opere diretamente sobre os espectrogramas dos áudios. Essa abordagem permite que o modelo aprenda as features relevantes automaticamente, em vez de depender da extração manual de MFCCs.

Utilizar Arquiteturas State-of-the-Art: Experimentar com modelos pré-treinados para verificação de locutor, como os baseados na arquitetura Transformer, que são o estado da arte para tarefas envolvendo dados sequenciais como a voz.
