import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia INFO e WARNING do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desliga oneDNN

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import pyttsx3
import time
import urllib.request


# --- FUNÇÃO DE NORMALIZAÇÃO GEOMÉTRICA (DEVE SER IGUAL À DA COLETA) ---
def normalizar_pontos_mao(mao_landmarks):
    """
    Subtrai o pulso (ponto 0) de todos os pontos e divide pela escala
    baseada nos eixos X e Y para evitar distorções do eixo Z.
    """
    pontos = np.array([[lm.x, lm.y, lm.z] for lm in mao_landmarks])

    # 1. Centralização: O pulso vira o ponto (0,0,0)
    pulso = pontos[0]
    pontos_centralizados = pontos - pulso

    # 2. Escalonamento corrigido: Calcula a escala baseando-se apenas em X e Y
    # Isso evita que o Z esmague ou distorça as coordenadas da tela.
    escala = np.max(np.abs(pontos_centralizados[:, :2]))  # Pega apenas colunas 0 e 1 (X e Y)
    if escala == 0:
        escala = 1

    pontos_normalizados = pontos_centralizados / escala
    return pontos_normalizados.flatten().tolist()


# ---------------------------------------------------------------------

# 1. Inicializar o motor de voz (Text-to-Speech)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidade da fala

# 2. Carregar o modelo treinado (Lembre-se de usar o modelo gerado pelo novo script SVM!)
try:
    modelo = joblib.load("modelo_libras1.pkl")  # Ajustado para o nome padrão salvo pelo novo treino
    print("💾 Modelo de classificação SVM carregado com sucesso!")
except Exception as e:
    print("❌ Erro ao carregar o modelo. Garanta que rodou o treinamento!")
    exit()

# 3. Configurar Novo MediaPipe Tasks
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("📥 Baixando arquivo de detecção do MediaPipe (Hand Landmarker)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("✅ Download concluído!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Variáveis de controle da frase e do tempo
frase_atual = ""
letra_anterior = ""
tempo_inicio_pose = None
TEMPO_PARA_CONFIRMAR = 2.0  # Segundos que a mão precisa ficar parada na pose

cap = cv2.VideoCapture(0)

print("\n📝 Formador de Frases iniciado!")
print("=========================================")
print("-> Segure a pose por 2 segundos para digitar a letra.")
print("-> Aperte 'ESPAÇO' para ouvir a frase em áudio.")
print("-> Aperte 'BACKSPACE' para apagar a última letra.")
print("-> Aperte 'Q' para encerrar.")
print("=========================================\n")

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Executa a detecção
    resultado = detector.detect(mp_image)

    letra_detectada = None
    pontos_normalizados = []

    # Se encontrou alguma mão
    if resultado.hand_landmarks:
        # Acessa explicitamente a primeira mão detectada da lista
        mao_landmarks = resultado.hand_landmarks[0]

        # Desenha os pontos na tela usando OpenCV puro
        h, w, _ = frame.shape
        for landmark in mao_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # 1. PROCESSA E NORMALIZA OS PONTOS GEOMETRICAMENTE
        pontos_normalizados = normalizar_pontos_mao(mao_landmarks)

        # 2. SE COMPLETO, PASSA PARA O MODELO SVM PREDIZER
        if len(pontos_normalizados) == 63:
            # Mandamos os dados normalizados como uma lista dentro de outra [pontos] para simular uma matriz
            letra_detectada = modelo.predict([pontos_normalizados])[0]

    # --- LÓGICA DO CRONÔMETRO DE CONFIRMAÇÃO ---
    if letra_detectada:
        if letra_detectada != letra_anterior:
            letra_anterior = letra_detectada
            tempo_inicio_pose = time.time()
        else:
            tempo_decorrido = time.time() - tempo_inicio_pose
            progresso = min(int((tempo_decorrido / TEMPO_PARA_CONFIRMAR) * 200), 200)
            cv2.rectangle(frame, (20, 110), (20 + progresso, 120), (0, 255, 255), -1)
            cv2.putText(frame, f"Segure... {tempo_decorrido:.1f}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if tempo_decorrido >= TEMPO_PARA_CONFIRMAR:
                frase_atual += letra_detectada
                print(f"🔤 Letra adicionada: {letra_detectada} | Frase atual: {frase_atual}")
                tempo_inicio_pose = time.time()
                letra_anterior = ""
                print('\a')  # Feedback sonoro
    else:
        letra_anterior = ""
        tempo_inicio_pose = None

    # --- INTERFACE VISUAL DA TELA ---
    cv2.rectangle(frame, (0, 0), (640, 70), (0, 0, 0), -1)
    letra_tela = letra_detectada if letra_detectada else "..."
    cv2.putText(frame, f"Sinal: {letra_tela}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Texto: {frase_atual}", (250, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Libras para Audio", frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord(' '):
        if frase_atual.strip():
            print(f"🔊 Falando: {frase_atual}")
            engine.say(frase_atual)
            engine.runAndWait()

    elif tecla == 8:  # Backspace
        frase_atual = frase_atual[:-1]
        print(f"⬅️ Letra apagada. Frase atual: {frase_atual}")

    elif tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
