import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Trocamos o RandomForestClassifier por SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Carregar e juntar os dados da pasta
X = []  # Landmarks
y = []  # Letras (Etiquetas)

print("📦 Carregando arquivos de dados...")
arquivos_csv = glob.glob("dados/*.csv")

if not arquivos_csv:
    print("❌ Erro: Nenhum arquivo .csv encontrado na pasta 'dados'. Faça a coleta primeiro!")
    exit()

for arquivo_csv in arquivos_csv:
    letra = os.path.basename(arquivo_csv).replace(".csv", "")
    # Ignora a primeira linha que é o cabeçalho (X0, Y0...)
    df = pd.read_csv(arquivo_csv)

    for _, linha in df.iterrows():
        X.append(linha.values)
        y.append(letra)

X = np.array(X)
y = np.array(y)

print(f"📊 Total de fotos carregadas: {len(X)} divididas entre as letras: {np.unique(y)}")

# 2. Separar os dados: 80% para treino e 20% para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧠 Treinando o modelo...")

# 3. Treinar o classificador SVM (Substituindo o Random Forest)
# C=10.0 ajuda o modelo a focar melhor em detalhes rígidos de posicionamento dos dedos
modelo = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
modelo.fit(X_treino, y_treino)

# 4. Avaliar o modelo com os 20% de dados reservados para teste
previsoes = modelo.predict(X_teste)
precisao = accuracy_score(y_teste, previsoes)

print("\n🎉 Treinamento concluído com sucesso!")
print(f"🎯 Precisão Geral do Modelo com SVM: {precisao * 100:.2f}%")
print("\n📝 Relatório de desempenho por letra:")
print(classification_report(y_teste, previsoes))

# 5. Salvar o modelo treinado para usarmos na webcam depois
joblib.dump(modelo, "modelo_libras1.pkl")
print("💾 Modelo SVM salvo com sucesso!")
