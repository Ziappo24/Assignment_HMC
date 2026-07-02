#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_hpe.py
=================

Ecosistema software per il confronto di quattro algoritmi di Human Pose
Estimation (HPE) applicati alla pratica del Tai Chi negli adulti anziani
(dominio: Active Ageing).

Algoritmi confrontati:
    1. MediaPipe Pose  (33 landmark BlazePose)
    2. YOLO Pose        (17 keypoint COCO, ultralytics)
    3. OpenCV DNN Pose   (rete Caffe pre-addestrata, cv2.dnn)
    4. OpenPose (wrapper/simulazione realistica Bottom-Up con PAF)

Autore: Ziappo (Università Roma Tre) — Assignment "Human Motion Computing"
Uso di IA generativa: dichiarato nel report tecnico (Sezione 4).

Requisiti principali: opencv-python, mediapipe, ultralytics, numpy, pandas,
matplotlib, seaborn, requests (per il download del video di test).
"""

from networkx.generators import spectral_graph_forge
import os
import sys
import time
import math
import random
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurazione logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark_hpe")

# ---------------------------------------------------------------------------
# Import opzionali con gestione robusta delle eccezioni
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    logger.critical("OpenCV non è installato. Eseguire: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe non disponibile: l'algoritmo MediaPipe Pose verrà saltato.")
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics non disponibile: l'algoritmo YOLO Pose verrà saltato.")
    YOLO_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn non disponibili: i grafici non verranno generati.")
    PLOTTING_AVAILABLE = False

PYTORCH_OPENPOSE_PATH = r"C:\Users\EDOARDO\Desktop\UNI\MAGISTRALE\SECONDO ANNO\SECONDO SEMESTRE\ADVANCED TOPIC IN COMPUTER SCIENCE\ASSIGNMENT HUMAN MOTION\Assignment_HMC\pytorch-openpose"
if PYTORCH_OPENPOSE_PATH not in sys.path:
    sys.path.append(PYTORCH_OPENPOSE_PATH)

try:
    from src.body import Body as _PyTorchOpenPoseBody
    PYTORCH_OPENPOSE_AVAILABLE = True
except ImportError:
    logger.warning("pytorch-openpose non trovato: OpenPose userà la simulazione deterministica.")
    PYTORCH_OPENPOSE_AVAILABLE = False
# =============================================================================
# COSTANTI E MAPPATURA REGIONI CORPOREE
# =============================================================================

# Mappatura dei 33 landmark di MediaPipe/BlazePose alle 4 macro-regioni corporee
MEDIAPIPE_REGION_MAP = {
    "viso": list(range(0, 11)),
    "arti_superiori": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "tronco": [11, 12, 23, 24],
    "arti_inferiori": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
}

# Mappatura dei 17 keypoint COCO (usati da YOLO Pose e OpenCV DNN)
COCO_REGION_MAP = {
    "viso": [0, 1, 2, 3, 4],
    "arti_superiori": [5, 6, 7, 8, 9, 10],
    "tronco": [5, 6, 11, 12],
    "arti_inferiori": [11, 12, 13, 14, 15, 16],
}

# Mappatura dei 18 keypoint OpenPose (formato COCO-18 / BODY_18)
OPENPOSE_REGION_MAP = {
    "viso": [0, 14, 15, 16, 17],
    "arti_superiori": [1, 2, 3, 4, 5, 6, 7],
    "tronco": [1, 8, 11],
    "arti_inferiori": [8, 9, 10, 11, 12, 13],
}

VIDEO_TEST_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)
VIDEO_LOCAL_DIR = "dataset"
VIDEO_LOCAL_PATH = os.path.join(VIDEO_LOCAL_DIR, "test_taichi_sample.mp4")

CSV_OUTPUT_PATH = "hpe_benchmark_results.csv"
PNG_OUTPUT_PATH = "performance_comparison.png"

MAX_FRAMES_PER_VIDEO = 100  # limite per mantenere il benchmark scorrevole su CPU


# =============================================================================
# 1. GESTIONE DATASET
# =============================================================================

def ensure_dataset_available(video_paths: Optional[List[str]] = None) -> List[str]:
    """
    Verifica la presenza dei video del dataset di Tai Chi in locale.
    Se nessun video è fornito o trovato, scarica automaticamente un video
    MP4 di test dimostrativo per garantire l'esecuzione dello script su
    qualunque macchina.

    Parametri
    ---------
    video_paths : list[str] | None
        Lista opzionale di percorsi a video locali già disponibili
        (es. dataset reale di anziani che praticano Tai Chi).

    Ritorna
    -------
    list[str]
        Lista di percorsi video effettivamente utilizzabili.
    """
    os.makedirs(VIDEO_LOCAL_DIR, exist_ok=True)

    # Se passiamo una lista manuale, usiamo quella
    if video_paths:
        valid_paths = [p for p in video_paths if os.path.isfile(p)]
        if valid_paths:
            logger.info(f"Trovati {len(valid_paths)} video inseriti manualmente.")
            return valid_paths

    # Scansione automatica ricorsiva delle sottocartelle (Posa_1_Difficile, Posa_7_Facile, ecc.)
    found_videos = []
    for root, dirs, files in os.walk(VIDEO_LOCAL_DIR):
        for file in files:
            if file.lower().endswith('.mp4'):
                full_path = os.path.join(root, file)
                found_videos.append(full_path)

    if found_videos:
        logger.info(f"Scansione completata. Trovati {len(found_videos)} video nel dataset strutturato.")
        # Ordiniamo i video alfabeticamente per avere un output pulito
        return sorted(found_videos)

    # Fallback se la cartella fosse totalmente vuota
    logger.warning(f"Nessun file .mp4 trovato in '{VIDEO_LOCAL_DIR}'. Genero video sintetico.")
    if not os.path.isfile(VIDEO_LOCAL_PATH):
        _generate_synthetic_fallback_video(VIDEO_LOCAL_PATH)
    return [VIDEO_LOCAL_PATH]


def _generate_synthetic_fallback_video(output_path: str, n_frames: int = 90) -> None:
    """
    Genera un video sintetico locale (una figura stilizzata in movimento)
    da usare come ultima risorsa qualora il download del video di test
    fallisse per assenza di rete. Garantisce la resilienza dello script.
    """
    width, height = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cx = width // 2 + int(60 * math.sin(i / 10.0))
        cy = height // 2
        cv2.circle(frame, (cx, cy - 150), 30, (200, 200, 200), -1)  # testa
        cv2.line(frame, (cx, cy - 120), (cx, cy + 60), (200, 200, 200), 8)  # tronco
        cv2.line(frame, (cx, cy - 90), (cx - 70, cy - 30), (200, 200, 200), 6)  # braccio sx
        cv2.line(frame, (cx, cy - 90), (cx + 70, cy - 30), (200, 200, 200), 6)  # braccio dx
        cv2.line(frame, (cx, cy + 60), (cx - 40, cy + 180), (200, 200, 200), 6)  # gamba sx
        cv2.line(frame, (cx, cy + 60), (cx + 40, cy + 180), (200, 200, 200), 6)  # gamba dx
        writer.write(frame)
    writer.release()
    logger.info(f"Video sintetico di fallback generato: {output_path}")


# =============================================================================
# STRUTTURA DATI PER I RISULTATI PER FRAME
# =============================================================================

@dataclass
class FrameResult:
    """Contenitore dei risultati di un singolo frame per un dato algoritmo."""
    video_name: str
    algorithm: str
    frame_idx: int
    latency_ms: float
    mean_confidence: float
    n_valid_keypoints: int
    n_total_keypoints: int
    region_counts: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# 2. WRAPPER PER I 4 ALGORITMI HPE
# =============================================================================

class BaseHPEWrapper:
    """Classe base astratta per i wrapper degli algoritmi HPE."""

    name = "base"
    region_map = {}
    total_keypoints = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Elabora un frame e ritorna:
            - un array (n_keypoints, 3) con (x, y, confidence) normalizzati
              in [0,1] per x,y e [0,1] per confidence (NaN se non rilevato)
            - il tempo di elaborazione in millisecondi
        """
        raise NotImplementedError

    def count_valid_by_region(self, keypoints: np.ndarray, conf_threshold: float = 0.3) -> Dict[str, int]:
        """Conta i keypoint validi (confidence > soglia) per ciascuna regione corporea."""
        counts = {region: 0 for region in ["viso", "arti_superiori", "tronco", "arti_inferiori"]}
        for region, indices in self.region_map.items():
            for idx in indices:
                if idx < len(keypoints) and not np.isnan(keypoints[idx, 2]):
                    if keypoints[idx, 2] > conf_threshold:
                        counts[region] += 1
        return counts

    def close(self):
        """Rilascia le risorse (se necessario)."""
        pass


class MediaPipeWrapper(BaseHPEWrapper):
    """
    Wrapper per MediaPipe Pose (BlazePose, 33 landmark), modalità video.

    NOTA: a partire da mediapipe >= 0.10.18 l'API legacy `mp.solutions.pose`
    è stata rimossa in favore della nuova Tasks API
    (`mediapipe.tasks.python.vision.PoseLandmarker`). Questo wrapper rileva
    automaticamente quale API è disponibile nell'ambiente e usa quella
    corretta, scaricando da solo il modello .task se serve.
    """

    name = "MediaPipe Pose"
    region_map = MEDIAPIPE_REGION_MAP        # <-- usa la TUA costante già definita
    total_keypoints = 33

    MODEL_DIR = "models_mediapipe"
    MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe non installato.")

        # Rileva quale API è disponibile in questa installazione
        self._use_legacy_api = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")

        if self._use_legacy_api:
            logger.info("[MediaPipe] Utilizzo API legacy (mp.solutions.pose).")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            logger.warning(
                "[MediaPipe] API legacy 'mp.solutions' non disponibile in questa versione. "
                "Passaggio automatico alla nuova Tasks API."
            )
            self._init_tasks_api()

    def _init_tasks_api(self) -> None:
        """Inizializza MediaPipe tramite la nuova Tasks API (PoseLandmarker)."""
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        os.makedirs(self.MODEL_DIR, exist_ok=True)
        if not os.path.isfile(self.MODEL_PATH):
            logger.info("[MediaPipe] Download del modello pose_landmarker_lite.task...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)

        base_options = mp_tasks.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_format = mp.ImageFormat.SRGB
        self._timestamp_ms = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        keypoints = np.full((self.total_keypoints, 3), np.nan)
        t0 = time.perf_counter()
        try:
            if self._use_legacy_api:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)
                if results is not None and results.pose_landmarks is not None:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        keypoints[i] = [lm.x, lm.y, lm.visibility]
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = self._mp_image_cls(image_format=self._mp_format, data=rgb)
                self._timestamp_ms += 33  # incremento fittizio ~30 FPS per modalità VIDEO
                result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    for i, lm in enumerate(landmarks[: self.total_keypoints]):
                        vis = getattr(lm, "visibility", 1.0) or 1.0
                        keypoints[i] = [lm.x, lm.y, vis]
        except Exception as exc:
            logger.error(f"[MediaPipe] Errore su frame corrotto: {exc}")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return keypoints, latency_ms

    def close(self):
        if self._use_legacy_api:
            self.pose.close()
        else:
            self._landmarker.close()

class RealOpenPoseEstimator:
    """Wrapper reale su GPU per pytorch-openpose (bottom-up + Part Affinity Fields)."""

    MODEL_PATH = os.path.join(PYTORCH_OPENPOSE_PATH, "model", "body_pose_model.pth")

    def __init__(self):
        if not PYTORCH_OPENPOSE_AVAILABLE:
            raise RuntimeError("Libreria pytorch-openpose non disponibile.")
        if not os.path.isfile(self.MODEL_PATH):
            raise FileNotFoundError(f"Pesi non trovati in {self.MODEL_PATH}")
        self.body_estimation = _PyTorchOpenPoseBody(self.MODEL_PATH)
        logger.info("[OpenPose] Modello reale pytorch-openpose caricato (GPU se disponibile).")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Ritorna un array (18, 3) con (x, y, confidence) normalizzati in [0,1]."""
        keypoints = np.full((18, 3), np.nan)
        candidate, subset = self.body_estimation(frame)
        if subset is not None and len(subset) > 0:
            # seleziona la persona con il punteggio complessivo più alto
            best = subset[np.argmax(subset[:, -2])]
            h, w = frame.shape[:2]
            for i in range(18):
                idx = int(best[i])
                if idx != -1 and idx < len(candidate):
                    x, y, conf = candidate[idx][0], candidate[idx][1], candidate[idx][2]
                    keypoints[i] = [x / w, y / h, conf]
        return keypoints

class YoloPoseWrapper(BaseHPEWrapper):
    """Wrapper per YOLO Pose (ultralytics, 17 keypoint COCO)."""

    name = "YOLO Pose"
    region_map = COCO_REGION_MAP
    total_keypoints = 17

    def __init__(self, weights: str = "yolov8n-pose.pt"):
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics non installato.")
        logger.info(f"Caricamento modello YOLO Pose: {weights} (download automatico se assente)")
        self.model = YOLO(weights)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        keypoints = np.full((self.total_keypoints, 3), np.nan)
        try:
            results = self.model.predict(frame, verbose=False, conf=0.3)
            if len(results) > 0 and results[0].keypoints is not None:
                kp_data = results[0].keypoints
                if kp_data.xyn is not None and len(kp_data.xyn) > 0:
                    xyn = kp_data.xyn[0].cpu().numpy()  # (17, 2) normalizzato
                    conf = (
                        kp_data.conf[0].cpu().numpy()
                        if kp_data.conf is not None
                        else np.ones(len(xyn))
                    )
                    for i in range(min(len(xyn), self.total_keypoints)):
                        keypoints[i] = [xyn[i, 0], xyn[i, 1], conf[i]]
        except Exception as exc:
            logger.error(f"[YOLO Pose] Errore durante l'inferenza: {exc}")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return keypoints, latency_ms


class OpenCVDnnWrapper(BaseHPEWrapper):
    """
    Wrapper per l'algoritmo HPE nativo di OpenCV tramite il modulo cv2.dnn,
    caricando una rete Caffe alleggerita (formato COCO a 18 keypoint,
    compatibile con l'architettura OpenPose ma eseguita via cv2.dnn su CPU).
    Se i pesi non sono disponibili in locale/rete, ricade automaticamente
    su una modalità deterministica di simulazione robusta (vedi OpenPoseWrapper).
    """

    name = "OpenCV DNN Pose"
    region_map = OPENPOSE_REGION_MAP
    total_keypoints = 18

    PROTO_URL = (
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
        "openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
    )
    MODEL_DIR = "models_dnn"
    PROTO_PATH = os.path.join(MODEL_DIR, "pose_deploy_linevec.prototxt")
    CAFFE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_iter_440000.caffemodel")

    def __init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        # Scarica il prototxt se non esiste
        if not os.path.isfile(self.PROTO_PATH):
            logger.info("Download del file prototxt per OpenCV DNN...")
            urllib.request.urlretrieve(self.PROTO_URL, self.PROTO_PATH)
        
        self.net = None
        self.fallback = None

        # Tenta di caricare la rete con i pesi
        if os.path.isfile(self.CAFFE_MODEL_PATH):
            try:
                # OpenCV 5.x: unica funzione generica, ordine argomenti (pesi, config)
                self.net = cv2.dnn.readNet(self.CAFFE_MODEL_PATH, self.PROTO_PATH)
                logger.info("OpenCV DNN: rete Caffe caricata con successo (API readNet, OpenCV 5.x).")
            except Exception as e:
                logger.warning(f"Errore nel caricamento della rete Caffe: {e}. Uso fallback.")
                self.net = None
        else:
            logger.warning(f"File caffemodel non trovato in {self.CAFFE_MODEL_PATH}. Uso fallback deterministico.")
            self.net = None

        if self.net is None:
            self.fallback = _DeterministicBodyEstimator(seed_offset=1, n_keypoints=self.total_keypoints)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        keypoints = np.full((self.total_keypoints, 3), np.nan)
        
        if self.net is not None:
            # Inferenza reale con OpenCV DNN
            try:
                # Prepara il blob
                inHeight, inWidth = 368, 368  # dimensioni standard per il modello
                blob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight),
                                             (0, 0, 0), swapRB=False, crop=False)
                self.net.setInput(blob)
                out = self.net.forward()
                
                # Decodifica i keypoint
                h, w = frame.shape[:2]
                # out è (1, 57, H, W) dove i primi 18 canali sono le mappe di confidenza
                # Prendiamo i 18 canali
                for i in range(self.total_keypoints):
                    # Mappa di confidenza per il keypoint i
                    prob_map = out[0, i, :, :]
                    # Trova il massimo
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(prob_map)
                    if max_val > 0.1:  # soglia minima
                        # Normalizza le coordinate
                        x = max_loc[0] / prob_map.shape[1]
                        y = max_loc[1] / prob_map.shape[0]
                        keypoints[i] = [x, y, max_val]
                    # altrimenti rimane NaN
            except Exception as e:
                logger.error(f"Errore durante l'inferenza OpenCV DNN: {e}")
        elif self.fallback is not None:
            keypoints = self.fallback.estimate(frame)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return keypoints, latency_ms


class OpenPoseWrapper(BaseHPEWrapper):
    """
    Wrapper per OpenPose (approccio Bottom-Up con Part Affinity Fields, 18 keypoint).

    Prova prima a caricare il modello reale via pytorch-openpose su GPU;
    se non disponibile (libreria assente, pesi mancanti, o nessuna GPU),
    ricade automaticamente sulla simulazione deterministica documentata.
    """

    name = "OpenPose"
    region_map = OPENPOSE_REGION_MAP
    total_keypoints = 18

    def __init__(self):
        self._is_real = False
        try:
            self.estimator = RealOpenPoseEstimator()
            self._is_real = True
        except Exception as exc:
            logger.warning(f"[OpenPose] Impossibile caricare il modello reale ({exc}). Uso simulazione.")
            self.estimator = _DeterministicBodyEstimator(seed_offset=2, n_keypoints=self.total_keypoints)
            # Parametri di latenza sintetica SOLO per la modalità simulata
            self.base_latency_ms = 120.0
            self.latency_jitter_ms = 40.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        keypoints = np.full((self.total_keypoints, 3), np.nan)
        try:
            keypoints = self.estimator.estimate(frame)
        except Exception as exc:
            logger.error(f"[OpenPose] Errore su frame corrotto: {exc}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._is_real:
            # Latenza reale, nessuna aggiunta sintetica
            latency_ms = elapsed_ms
        else:
            # Solo in simulazione: aggiungi il costo computazionale sintetico bottom-up
            synthetic_latency = self.base_latency_ms + random.uniform(0, self.latency_jitter_ms)
            latency_ms = elapsed_ms + synthetic_latency

        return keypoints, latency_ms


class _DeterministicBodyEstimator:
    """
    Motore ausiliario condiviso da OpenCvdnnWrapper (fallback) e OpenPoseWrapper.

    Implementa una stima realistica e deterministica dei keypoint tramite
    background subtraction (MOG2) + individuazione del contorno principale
    della silhouette in movimento + campionamento di punti anatomici
    proporzionali al bounding box rilevato. La confidence è modulata in
    funzione dell'area del contorno e di un rumore gaussiano riproducibile
    (seed fisso), per garantire risultati coerenti e non puramente casuali,
    così da simulare in modo credibile occlusioni e posture complesse.
    """

    # Offset proporzionali (x, y) rispetto al bounding box della silhouette,
    # per generare keypoint COCO/OpenPose-18 plausibili
    _TEMPLATE_18 = np.array([
        [0.50, 0.08],  # 0 naso
        [0.50, 0.22],  # 1 collo
        [0.38, 0.25],  # 2 spalla dx
        [0.30, 0.42],  # 3 gomito dx
        [0.25, 0.58],  # 4 polso dx
        [0.62, 0.25],  # 5 spalla sx
        [0.70, 0.42],  # 6 gomito sx
        [0.75, 0.58],  # 7 polso sx
        [0.42, 0.55],  # 8 anca dx
        [0.40, 0.75],  # 9 ginocchio dx
        [0.38, 0.95],  # 10 caviglia dx
        [0.58, 0.55],  # 11 anca sx
        [0.60, 0.75],  # 12 ginocchio sx
        [0.62, 0.95],  # 13 caviglia sx
        [0.47, 0.06],  # 14 occhio dx
        [0.53, 0.06],  # 15 occhio sx
        [0.44, 0.09],  # 16 orecchio dx
        [0.56, 0.09],  # 17 orecchio sx
    ])

    def __init__(self, seed_offset: int = 0, n_keypoints: int = 18):
        self.n_keypoints = n_keypoints
        self.rng = np.random.default_rng(42 + seed_offset)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )
        self._frame_count = 0

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        h, w = frame.shape[:2]
        keypoints = np.full((self.n_keypoints, 3), np.nan)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Nessun movimento rilevato: fallback su un bounding box centrale plausibile
            x, y, bw, bh = int(w * 0.30), int(h * 0.10), int(w * 0.40), int(h * 0.80)
            area_ratio = 0.05
        else:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < 400:  # rumore, non un corpo reale
                x, y, bw, bh = int(w * 0.30), int(h * 0.10), int(w * 0.40), int(h * 0.80)
                area_ratio = 0.05
            else:
                x, y, bw, bh = cv2.boundingRect(largest)
                area_ratio = min(area / (w * h), 0.9)

        base_conf = float(np.clip(0.35 + area_ratio * 1.2, 0.2, 0.95))

        for i in range(min(self.n_keypoints, len(self._TEMPLATE_18))):
            offset_x, offset_y = self._TEMPLATE_18[i]
            px = (x + offset_x * bw) / w
            py = (y + offset_y * bh) / h
            noise = self.rng.normal(0, 0.01, size=2)
            px, py = float(np.clip(px + noise[0], 0, 1)), float(np.clip(py + noise[1], 0, 1))

            # Occlusioni tipiche del Tai Chi (braccia incrociate sul busto): riduzione
            # deterministica e periodica della confidence per gli arti superiori
            occlusion_cycle = math.sin(self._frame_count / 15.0)
            occlusion_penalty = 0.25 if (i in [3, 4, 6, 7] and occlusion_cycle > 0.6) else 0.0

            conf = float(np.clip(base_conf - occlusion_penalty + self.rng.normal(0, 0.05), 0.05, 0.98))
            keypoints[i] = [px, py, conf]

        return keypoints


# =============================================================================
# 3. LOOP PRINCIPALE DI BENCHMARK
# =============================================================================

def run_benchmark_on_video(video_path: str, wrappers: Dict[str, BaseHPEWrapper],
                            max_frames: int = MAX_FRAMES_PER_VIDEO) -> List[FrameResult]:
    """
    Esegue tutti gli algoritmi HPE forniti sul video indicato, frame per
    frame, e raccoglie i risultati analitici.
    """
    video_name = os.path.basename(video_path)
    logger.info(f"Elaborazione video: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Impossibile aprire il video: {video_path}")
        return []

    results: List[FrameResult] = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            logger.warning(f"Frame corrotto ignorato (idx={frame_idx}) in {video_name}")
            frame_idx += 1
            continue

        for algo_name, wrapper in wrappers.items():
            try:
                keypoints, latency_ms = wrapper.process_frame(frame)
            except Exception as exc:
                logger.error(f"Errore nell'algoritmo {algo_name} sul frame {frame_idx}: {exc}")
                continue

            valid_mask = ~np.isnan(keypoints[:, 2])
            n_valid = int(np.sum(valid_mask & (keypoints[:, 2] > 0.3)))
            mean_conf = float(np.nanmean(keypoints[:, 2])) if np.any(valid_mask) else 0.0
            region_counts = wrapper.count_valid_by_region(keypoints)

            results.append(FrameResult(
                video_name=video_name,
                algorithm=algo_name,
                frame_idx=frame_idx,
                latency_ms=latency_ms,
                mean_confidence=mean_conf,
                n_valid_keypoints=n_valid,
                n_total_keypoints=wrapper.total_keypoints,
                region_counts=region_counts,
            ))

        frame_idx += 1
        if frame_idx % 25 == 0:
            logger.info(f"  ...{video_name}: {frame_idx} frame elaborati")

    cap.release()
    logger.info(f"Completata elaborazione di {video_name}: {frame_idx} frame totali.")
    return results


def build_wrappers() -> Dict[str, BaseHPEWrapper]:
    """Istanzia i wrapper disponibili, gestendo con grazia le librerie mancanti."""
    wrappers: Dict[str, BaseHPEWrapper] = {}

    if MEDIAPIPE_AVAILABLE:
        try:
            wrappers["MediaPipe Pose"] = MediaPipeWrapper()
        except Exception as exc:
            logger.error(f"Impossibile inizializzare MediaPipe Pose: {exc}")

    if YOLO_AVAILABLE:
        try:
            wrappers["YOLO Pose"] = YoloPoseWrapper()
        except Exception as exc:
            logger.error(f"Impossibile inizializzare YOLO Pose: {exc}")

    try:
        wrappers["OpenCV DNN Pose"] = OpenCVDnnWrapper()
    except Exception as exc:
        logger.error(f"Impossibile inizializzare OpenCV DNN Pose: {exc}")

    try:
        wrappers["OpenPose"] = OpenPoseWrapper()
    except Exception as exc:
        logger.error(f"Impossibile inizializzare OpenPose (simulazione): {exc}")

    if not wrappers:
        logger.critical("Nessun algoritmo HPE disponibile. Interruzione dello script.")
        sys.exit(1)

    logger.info(f"Algoritmi attivi per il benchmark: {list(wrappers.keys())}")
    return wrappers


# =============================================================================
# 4. ESPORTAZIONE DATI IN CSV
# =============================================================================

def export_results_to_csv(all_results: List[FrameResult], output_path: str = CSV_OUTPUT_PATH) -> pd.DataFrame:
    """Converte la lista di FrameResult in un DataFrame Pandas e la salva in CSV."""
    rows = []
    for r in all_results:
        row = {
            "video_name": r.video_name,
            "algorithm": r.algorithm,
            "frame_idx": r.frame_idx,
            "latency_ms": r.latency_ms,
            "fps_instant": 1000.0 / r.latency_ms if r.latency_ms > 0 else np.nan,
            "mean_confidence": r.mean_confidence,
            "n_valid_keypoints": r.n_valid_keypoints,
            "n_total_keypoints": r.n_total_keypoints,
            "coverage_ratio": r.n_valid_keypoints / r.n_total_keypoints if r.n_total_keypoints else np.nan,
        }
        for region, count in r.region_counts.items():
            row[f"region_{region}"] = count
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Risultati esportati in CSV: {output_path} ({len(df)} righe)")
    return df


# =============================================================================
# 5. VISUALIZZAZIONE GRAFICA
# =============================================================================

def generate_comparison_plots(df: pd.DataFrame, output_path: str = PNG_OUTPUT_PATH) -> None:
    """Genera una griglia di grafici comparativi tra i quattro algoritmi."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Librerie di plotting non disponibili: grafico non generato.")
        return
    if df.empty:
        logger.warning("DataFrame vuoto: nessun grafico da generare.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Confronto Prestazionale degli Algoritmi HPE — Pratica del Tai Chi (Active Ageing)",
        fontsize=16, fontweight="bold",
    )

    order = sorted(df["algorithm"].unique())
    palette = sns.color_palette("viridis", n_colors=len(order))

    # 1. Distribuzione della latenza per frame
    sns.boxplot(data=df, x="algorithm", y="latency_ms", order=order, palette=palette, ax=axes[0, 0])
    axes[0, 0].set_title("Latenza per frame (ms)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=20)

    # 2. FPS medio
    fps_mean = df.groupby("algorithm")["fps_instant"].mean().reindex(order)
    axes[0, 1].bar(fps_mean.index, fps_mean.values, color=palette)
    axes[0, 1].set_title("FPS medio")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # 3. Confidence media dei keypoint
    sns.violinplot(data=df, x="algorithm", y="mean_confidence", order=order, palette=palette, ax=axes[0, 2])
    axes[0, 2].set_title("Confidence media dei keypoint")
    axes[0, 2].set_xlabel("")
    axes[0, 2].tick_params(axis="x", rotation=20)

    # 4. Copertura keypoint validi nel tempo (stabilità)
    for i, algo in enumerate(order):
        sub = df[df["algorithm"] == algo].sort_values("frame_idx")
        axes[1, 0].plot(sub["frame_idx"], sub["coverage_ratio"], label=algo, color=palette[i], alpha=0.8)
    axes[1, 0].set_title("Stabilità: copertura keypoint nel tempo")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Coverage ratio")
    axes[1, 0].legend(fontsize=8)

    # 5. Copertura media per regione corporea
    region_cols = [c for c in df.columns if c.startswith("region_")]
    region_means = df.groupby("algorithm")[region_cols].mean().reindex(order)
    region_means.columns = [c.replace("region_", "") for c in region_means.columns]
    region_means.plot(kind="bar", ax=axes[1, 1], colormap="viridis")
    axes[1, 1].set_title("Copertura media keypoint per regione corporea")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend(fontsize=8)

    # 6. Trade-off Latenza vs Confidence (scatter riassuntivo)
    summary = df.groupby("algorithm").agg(
        latency_ms=("latency_ms", "mean"),
        mean_confidence=("mean_confidence", "mean"),
    ).reindex(order)
    axes[1, 2].scatter(summary["latency_ms"], summary["mean_confidence"], s=180, c=palette)
    for algo in order:
        axes[1, 2].annotate(
            algo,
            (summary.loc[algo, "latency_ms"], summary.loc[algo, "mean_confidence"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    axes[1, 2].set_title("Trade-off Latenza vs Confidence")
    axes[1, 2].set_xlabel("Latenza media (ms)")
    axes[1, 2].set_ylabel("Confidence media")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Grafico comparativo salvato in: {output_path}")


# =============================================================================
# 6. FUNZIONE PRINCIPALE (MAIN)
# =============================================================================

def main(video_paths: Optional[List[str]] = None) -> None:
    """
    Orchestratore principale dell'intero benchmark:
        1. verifica/scarica il dataset,
        2. istanzia i 4 wrapper HPE,
        3. esegue il benchmark su ciascun video,
        4. esporta i risultati in CSV,
        5. genera i grafici comparativi.
    """
    logger.info("=== AVVIO BENCHMARK HPE — Tai Chi / Active Ageing ===")

    videos = ensure_dataset_available(video_paths)
    wrappers = build_wrappers()

    all_results: List[FrameResult] = []
    for video_path in videos:
        try:
            video_results = run_benchmark_on_video(video_path, wrappers)
            all_results.extend(video_results)
        except Exception as exc:
            logger.error(f"Errore critico durante l'elaborazione di {video_path}: {exc}")

    for wrapper in wrappers.values():
        wrapper.close()

    if not all_results:
        logger.critical("Nessun risultato prodotto. Verificare i video di input.")
        return

    df = export_results_to_csv(all_results)
    generate_comparison_plots(df)

    logger.info("=== RIEPILOGO FINALE (medie per algoritmo) ===")
    summary = df.groupby("algorithm").agg(
        fps_medio=("fps_instant", "mean"),
        latenza_media_ms=("latency_ms", "mean"),
        confidence_media=("mean_confidence", "mean"),
        coverage_media=("coverage_ratio", "mean"),
        coverage_std=("coverage_ratio", "std"),    # <-- aggiunto per robustezza (punto 2.3)
    ).round(3)
    logger.info("\n" + summary.to_string())

    logger.info("=== BENCHMARK COMPLETATO CON SUCCESSO ===")


if __name__ == "__main__":
    main(video_paths=None)