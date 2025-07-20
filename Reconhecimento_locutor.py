import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import random
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# pasta que contém os arquivos .tsv e a pasta 'clips'.
DATASET_PATH = './common_voice/pt' 

CLIPS_DIR = os.path.join(DATASET_PATH, 'clips')
MIN_CLIPS_PER_SPEAKER = 4 
NUM_PAIRS_PER_SPEAKER = 10


def extract_mfcc(file_path, n_mfcc=20, max_len=174):
    """
    Extrai os coeficientes MFCC de um arquivo de áudio.
    """
    try:
        if not os.path.exists(file_path):
            return None
        audio, sample_rate = librosa.load(file_path, sr=None, duration=5.0) # Limita a 5s para velocidade
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        if mfccs.shape[1] > max_len:
            mfccs = mfccs[:, :max_len]
        else:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return np.mean(mfccs, axis=1)
    except Exception as e:
        # print(f"Aviso: Erro ao processar {file_path}: {e}")
        return None

# --- PASSO 1: CARREGAR E UNIR TODOS OS DADOS ---

print("Procurando por todos os arquivos de metadados (.tsv)...")
all_tsv_files = glob.glob(os.path.join(DATASET_PATH, '*.tsv'))

files_to_load = [f for f in all_tsv_files if any(name in f for name in ['validated.tsv', 'other.tsv', 'invalidated.tsv'])]

if not files_to_load:
    print(f"ERRO CRÍTICO: Nenhum arquivo de metadados (validated.tsv, other.tsv, etc.) foi encontrado em: {DATASET_PATH}")
    exit()

print(f"Carregando e unindo os seguintes arquivos: {files_to_load}")

df_list = []
for file in files_to_load:
    try:
        df_temp = pd.read_csv(file, sep='\t', on_bad_lines='skip')
        if 'client_id' in df_temp.columns and 'path' in df_temp.columns:
            df_list.append(df_temp[['client_id', 'path']])
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o arquivo {file}. Erro: {e}")

df = pd.concat(df_list, ignore_index=True)

print(f"Total de {len(df)} registros de áudio carregados de todos os arquivos.")

print("\n--- DIAGNÓSTICO DO DATASET (DADOS COMBINADOS) ---")
print("Distribuição dos locutores (Top 15 com mais áudios):")
print(df['client_id'].value_counts().head(15))
print("---------------------------------\n")


print(f"Filtrando locutores com no mínimo {MIN_CLIPS_PER_SPEAKER} áudios...")
speaker_counts = df['client_id'].value_counts()
valid_speakers = speaker_counts[speaker_counts >= MIN_CLIPS_PER_SPEAKER].index
df_valid = df[df['client_id'].isin(valid_speakers)].copy()

print(f"Encontrados {len(valid_speakers)} locutores válidos.")

if len(valid_speakers) < 10:
    print("\nAVISO IMPORTANTE:")
    print(f"Foram encontrados apenas {len(valid_speakers)} locutores. Este número ainda é baixo.")
    if len(valid_speakers) < 2:
        print("ERRO: Menos de 2 locutores encontrados. Impossível criar pares negativos. Saindo.")
        exit()


# --- PASSO 2: CRIAR PARES E EXTRAIR CARACTERÍSTICAS ---

features = []
labels = []

print("Criando pares e extraindo características (MFCCs)...")
speaker_groups = df_valid.groupby('client_id')['path'].apply(list)

for speaker_id, clips in tqdm(speaker_groups.items(), desc="Processando locutores"):
    
    # Pares Positivos
    num_positive_clips = min(len(clips), NUM_PAIRS_PER_SPEAKER * 2)
    if num_positive_clips % 2 != 0: num_positive_clips -= 1
        
    if num_positive_clips > 1:
        positive_pairs = random.sample(clips, num_positive_clips)
        for i in range(0, len(positive_pairs), 2):
            mfcc1 = extract_mfcc(os.path.join(CLIPS_DIR, positive_pairs[i]))
            mfcc2 = extract_mfcc(os.path.join(CLIPS_DIR, positive_pairs[i+1]))
            if mfcc1 is not None and mfcc2 is not None:
                features.append(np.abs(mfcc1 - mfcc2))
                labels.append(1)

    # Pares Negativos
    for _ in range(NUM_PAIRS_PER_SPEAKER):
        if not clips: continue
        clip1 = random.choice(clips)
        other_speaker_id = random.choice([sid for sid in valid_speakers if sid != speaker_id])
        clip2 = random.choice(speaker_groups[other_speaker_id])
        mfcc1 = extract_mfcc(os.path.join(CLIPS_DIR, clip1))
        mfcc2 = extract_mfcc(os.path.join(CLIPS_DIR, clip2))
        if mfcc1 is not None and mfcc2 is not None:
            features.append(np.abs(mfcc1 - mfcc2))
            labels.append(0)

# --- PASSO 3: PREPARAR DADOS ---

print("\nPreparando dados para os modelos...")
if not features:
    print("ERRO: Nenhuma característica foi extraída.")
    exit()

X = np.array(features)
y = np.array(labels)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dataset criado com {len(X)} amostras.")
print(f"Treinamento: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")

# --- PASSO 4: TREINAR E AVALIAR OS MODELOS ---

# Modelo k-NN
print("\n--- Treinando e Avaliando o k-NN ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Acurácia do k-NN:", accuracy_score(y_test, y_pred_knn))
report_knn = classification_report(y_test, y_pred_knn, target_names=['Diferente', 'Mesmo Locutor'], output_dict=True)
print(classification_report(y_test, y_pred_knn, target_names=['Diferente', 'Mesmo Locutor']))


# Modelo SVM
print("\n--- Treinando e Avaliando o SVM ---")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("Acurácia do SVM:", accuracy_score(y_test, y_pred_svm))
report_svm = classification_report(y_test, y_pred_svm, target_names=['Diferente', 'Mesmo Locutor'], output_dict=True)
print(classification_report(y_test, y_pred_svm, target_names=['Diferente', 'Mesmo Locutor']))


# --- PASSO 5: PLOTAR OS GRÁFICOS DE MÉTRICAS ---

def plot_model_results(report, model_name):
    """
    Plota as métricas de um modelo e salva o gráfico como uma imagem.
    """
    metrics_data = {
        'Diferente': report['Diferente'],
        'Mesmo Locutor': report['Mesmo Locutor'],
        'Média Ponderada': report['weighted avg']
    }
    
    for key in metrics_data:
        metrics_data[key].pop('support', None)
        
    df_metrics = pd.DataFrame(metrics_data).transpose()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = df_metrics.plot(kind='bar', figsize=(12, 7), colormap='viridis', rot=0)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

    plt.title(f'Desempenho do Modelo {model_name}', fontsize=16, weight='bold')
    plt.ylabel('Pontuação', fontsize=12)
    plt.xlabel('Classes e Média', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Métricas')
    plt.tight_layout()
    
    filename = f'{model_name}_results.png'
    plt.savefig(filename)
    print(f"\nGráfico de resultados para {model_name} salvo como '{filename}'")
    plt.close()

print("\n--- Gerando gráficos de desempenho ---")
plot_model_results(report_knn, 'k-NN')
plot_model_results(report_svm, 'SVM')


# --- PASSO 6: GERAR MATRIZ DE CONFUSÃO EM PNG ---

def plot_confusion_matrix_png(y_true, y_pred, model_name):
    """
    Calcula e plota a matriz de confusão como um heatmap e salva como PNG.
    """
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Extrai os valores da matriz de confusão: Verdadeiro Negativo, Falso Positivo, Falso Negativo, Verdadeiro Positivo
    tn, fp, fn, tp = cm.ravel()
    
    # Cria os rótulos para cada quadrante
    annot_labels = (np.asarray([f"VN\n{tn}", f"FP\n{fp}", f"FN\n{fn}", f"VP\n{tp}"])).reshape(2,2)
    
    # Labels para os eixos
    axis_labels = ['Diferente (Neg)', 'Mesmo Locutor (Pos)']
    
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
                     xticklabels=axis_labels, yticklabels=axis_labels, cbar=False)
    
    ax.set_xlabel('Valor Predito', fontsize=14, labelpad=20)
    ax.set_ylabel('Valor Real', fontsize=14, labelpad=20)
    ax.set_title(f'Matriz de Confusão: {model_name}\nAcurácia: {accuracy:.1%}', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    
    filename = f'{model_name}_confusion_matrix.png'
    plt.savefig(filename)
    print(f"Matriz de confusão para {model_name} salva como '{filename}'")
    plt.close()

print("\n--- Gerando Matrizes de Confusão (imagens PNG) ---")
plot_confusion_matrix_png(y_test, y_pred_knn, 'k-NN')
plot_confusion_matrix_png(y_test, y_pred_svm, 'SVM')
