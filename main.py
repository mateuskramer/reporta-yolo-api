"""
Reporta Criciúma — YOLO Detector API
Detecta buracos em imagens e vídeos usando modelo YOLOv8m treinado no MWPD.

Endpoints:
  GET  /health        → status da API e se o modelo está carregado
  POST /detect/image  → detecta buraco em imagem (png, jpg, webp)
  POST /detect/video  → detecta buraco em vídeo (mp4, mov, avi) via frame sampling
"""

import os
import cv2
import tempfile
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==============================================================
# CONFIGURAÇÃO
# ==============================================================

# Repositório do Hugging Face — configure via variáveis de ambiente no Render
HF_REPO_ID  = os.getenv("HF_REPO_ID",  "seu-usuario/reporta-yolo")
HF_FILENAME = os.getenv("HF_FILENAME", "checkpoint_epoch_190.pt")
HF_TOKEN    = os.getenv("HF_TOKEN")    # token de acesso (repo privado)

# Confiança mínima para considerar uma detecção válida
CONFIANCA_MINIMA = float(os.getenv("CONFIANCA_MINIMA", "0.30"))

# Para vídeo: amostra 1 frame a cada N segundos
FRAME_SAMPLE_INTERVAL = float(os.getenv("FRAME_SAMPLE_INTERVAL", "2.0"))

# Para vídeo: número mínimo de frames com detecção para confirmar buraco
# (evita falsos positivos de um frame isolado)
MIN_FRAMES_CONFIRMACAO = int(os.getenv("MIN_FRAMES_CONFIRMACAO", "2"))

# ==============================================================
# INICIALIZAÇÃO
# ==============================================================

app = FastAPI(
    title="Reporta Criciúma — YOLO API",
    description="Detecção de buracos via YOLOv8m (MWPD dataset)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir ao domínio do Streamlit
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Carrega o modelo uma vez ao iniciar (fica em memória)
model: Optional[YOLO] = None

@app.on_event("startup")
def carregar_modelo():
    global model
    print(f"📥 Baixando modelo do Hugging Face: {HF_REPO_ID}/{HF_FILENAME} ...")
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN,
        )
        print(f"🤖 Carregando modelo: {model_path} ...")
        model = YOLO(model_path)
        print("✅ Modelo carregado!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("A API sobe, mas /detect retornará erro até o modelo estar disponível.")

# ==============================================================
# SCHEMAS DE RESPOSTA
# ==============================================================

class DeteccaoImagem(BaseModel):
    detectou_buraco: bool
    confianca: float          # confiança máxima entre todas as detecções (0.0–1.0)
    n_deteccoes: int          # quantidade de bounding boxes na imagem
    mensagem: str

class DeteccaoVideo(BaseModel):
    detectou_buraco: bool
    confianca: float          # confiança média dos frames positivos
    n_frames_analisados: int
    n_frames_com_buraco: int
    mensagem: str

# ==============================================================
# UTILITÁRIOS
# ==============================================================

def _verificar_modelo():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo YOLO não carregado. Verifique se '{MODEL_PATH}' existe."
        )

def _detectar_em_frame(frame_bgr: np.ndarray) -> tuple[bool, float, int]:
    """
    Roda inferência em um frame (numpy array BGR).
    Retorna: (detectou, confianca_maxima, n_deteccoes)
    """
    results = model.predict(
        source=frame_bgr,
        conf=CONFIANCA_MINIMA,
        verbose=False,
        imgsz=640,
    )
    result = results[0]
    n = len(result.boxes)
    if n == 0:
        return False, 0.0, 0
    confs = result.boxes.conf.cpu().numpy()
    return True, float(confs.max()), n

# ==============================================================
# ENDPOINTS
# ==============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_carregado": model is not None,
        "modelo_path": MODEL_PATH,
        "confianca_minima": CONFIANCA_MINIMA,
    }


@app.post("/detect/image", response_model=DeteccaoImagem)
async def detectar_imagem(file: UploadFile = File(...)):
    """
    Recebe uma imagem e retorna se há buraco detectado.
    Aceita: image/jpeg, image/png, image/webp
    """
    _verificar_modelo()

    tipo = file.content_type or ""
    if not tipo.startswith("image/"):
        raise HTTPException(status_code=415, detail="Envie uma imagem (jpg, png, webp).")

    conteudo = await file.read()
    arr = np.frombuffer(conteudo, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Não foi possível decodificar a imagem.")

    detectou, confianca, n_det = _detectar_em_frame(frame)

    if detectou:
        msg = f"Buraco detectado com confiança {confianca:.0%}."
    else:
        msg = "Nenhum buraco detectado na imagem."

    return DeteccaoImagem(
        detectou_buraco=detectou,
        confianca=round(confianca, 3),
        n_deteccoes=n_det,
        mensagem=msg,
    )


@app.post("/detect/video", response_model=DeteccaoVideo)
async def detectar_video(file: UploadFile = File(...)):
    """
    Recebe um vídeo, amostra frames a cada FRAME_SAMPLE_INTERVAL segundos
    e retorna se há buraco detectado em frames suficientes.
    Aceita: video/mp4, video/quicktime, video/x-msvideo
    """
    _verificar_modelo()

    tipo = file.content_type or ""
    if not tipo.startswith("video/"):
        raise HTTPException(status_code=415, detail="Envie um vídeo (mp4, mov, avi).")

    conteudo = await file.read()

    # Salva em arquivo temporário porque cv2.VideoCapture precisa de path
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(conteudo)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Não foi possível abrir o vídeo.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        intervalo_frames = max(1, int(fps * FRAME_SAMPLE_INTERVAL))

        frames_analisados = 0
        frames_com_buraco = 0
        confiancas = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Amostra só frames no intervalo definido
            if frame_idx % intervalo_frames == 0:
                detectou, conf, _ = _detectar_em_frame(frame)
                frames_analisados += 1
                if detectou:
                    frames_com_buraco += 1
                    confiancas.append(conf)

            frame_idx += 1

        cap.release()

    finally:
        os.unlink(tmp_path)

    # Decisão final: exige mínimo de frames confirmando
    detectou_final = frames_com_buraco >= MIN_FRAMES_CONFIRMACAO
    confianca_media = float(np.mean(confiancas)) if confiancas else 0.0

    if detectou_final:
        msg = (f"Buraco confirmado em {frames_com_buraco}/{frames_analisados} frames "
               f"(confiança média: {confianca_media:.0%}).")
    elif frames_com_buraco > 0:
        msg = (f"Possível buraco em {frames_com_buraco} frame(s), "
               f"mas abaixo do limiar de confirmação ({MIN_FRAMES_CONFIRMACAO}).")
    else:
        msg = "Nenhum buraco detectado no vídeo."

    return DeteccaoVideo(
        detectou_buraco=detectou_final,
        confianca=round(confianca_media, 3),
        n_frames_analisados=frames_analisados,
        n_frames_com_buraco=frames_com_buraco,
        mensagem=msg,
    )
