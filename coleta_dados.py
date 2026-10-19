import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import urllib.request
import numpy as np  # Adicionado para os cálculos matemáticos


# --- FUNÇÃO DE NORMALIZAÇÃO GEOMÉTRICA ---
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


# ----------------------------------------

# 1. Configuração do MediaPipe (Tasks API)
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("📥 Baixando arquivo de detecção do MediaPipe (Hand Landmarker)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("✅ Download concluído!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Definir qual letra você vai coletar agora
letra_atual = input("Qual letra do alfabeto você vai coletar agora? (Ex: A, B, C): ").upper().strip()

# Criar a pasta 'dados' se ela não existir
if not os.path.exists("dados"):
    os.makedirs("dados")

caminho_csv = f"dados/{letra_atual}.csv"

# Se o arquivo não existir, cria e coloca o cabeçalho (X0, Y0, Z0... X20, Y20, Z20)
if not os.path.exists(caminho_csv):
    cabecalho = []
    for i in range(21):
        cabecalho.extend([f"X{i}", f"Y{i}", f"Z{i}"])
    with open(caminho_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(cabecalho)

# 3. Inicializar a Webcam
cap = cv2.VideoCapture(0)

# Descobrir quantas capturas já existem no arquivo
with open(caminho_csv, mode='r') as f:
    contador_capturas = sum(1 for linha in f) - 1  # Desconta a linha do cabeçalho

print(f"\n📸 Iniciando coleta para a letra {letra_atual}!")
print("=========================================")
print("-> Posicione a mão na tela fazendo o sinal.")
print("-> Aperte 'K' para tirar UMA foto.")
print("-> Mude a mão livremente de ângulo/distância e aperte 'K' de novo.")
print("-> Aperte 'Q' para salvar tudo e encerrar.")
print("=========================================\n")

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        print("Erro ao acessar a webcam.")
        break

    # Inverter o frame (efeito espelho) e converter para RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Processar os landmarks
    resultado = detector.detect(mp_image)
    pontos_para_salvar = []

    if resultado.hand_landmarks:
        mao_landmarks = resultado.hand_landmarks[0]

        # Desenha os pontos na tela usando OpenCV puro
        h, w, _ = frame.shape
        for landmark in mao_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # PROCESSA E NORMALIZA OS PONTOS AQUI
        pontos_para_salvar = normalizar_pontos_mao(mao_landmarks)

    # Informações na tela do OpenCV
    cv2.putText(frame, f"Letra: {letra_atual} | Fotos tiradas: {contador_capturas}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Coleta de Dados - 1 clique por vez", frame)

    tecla = cv2.waitKey(1) & 0xFF

    # SALVA APENAS UMA FOTO AO APERTAR 'K'
    if tecla == ord('k'):
        if len(pontos_para_salvar) == 63:
            with open(caminho_csv, mode='a', newline='') as f:
                csv.writer(f).writerow(pontos_para_salvar)
            contador_capturas += 1
            print(f"📸 Foto {contador_capturas} salva (Normalizada)!")
        else:
            print("⚠️ Erro: Mão não está visível ou incompleta na tela.")

    # FECHA O PROGRAMA AO APERTAR 'Q'
    elif tecla == ord('q'):
        break

# Finalização limpa
cap.release()
cv2.destroyAllWindows()
detector.close()

print(f"\n✅ Concluído! O arquivo 'dados/{letra_atual}.csv' agora tem {contador_capturas} fotos salvas.")
