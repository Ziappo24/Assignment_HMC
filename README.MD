"""# Benchmark di Algoritmi di Human Pose Estimation (HPE) per la Pratica del Tai Chi nell'Active Ageing

Questo repository contiene l'ecosistema software sviluppato per l'assignment del corso avanzato **"Human Motion Computing" (Advanced Topics in Computer Science)** tenuto dalla Prof.ssa Olga C. Santos presso l'**Università Roma Tre**.

Il progetto si colloca all'interno della **Parte II - Approccio B** del programma d'esame. L'obiettivo è analizzare, confrontare e valutare criticamente quattro diversi algoritmi di *Human Pose Estimation* (HPE) applicati al tracciamento e alla modellazione del movimento psicomotorio (*Physical User Modelling - PhyUM*) di soggetti adulti anziani durante l'esecuzione di esercizi di Tai Chi in un contesto domestico non controllato.

---

## 🚀 Architettura del Software e Algoritmi Analizzati

Il nucleo del sistema è lo script `benchmark_hpe.py`, un framework di test automatizzato e resiliente che scansiona ricorsivamente una cartella di dataset organizzata per cartelle di pose ed esegue l'estrazione e l'analisi dei landmark corporei frame-by-frame mediante:

1. **MediaPipe Pose**: Algoritmo basato sulla topologia *BlazePose* a 33 landmark geometrici, ottimizzato per l'esecuzione in tempo reale su dispositivi commerciali.
2. **YOLO Pose (Ultralytics)**: Rete neurale *state-of-the-art* basata sull'architettura *Top-Down* a 17 keypoint (standard COCO), celebre per la sua velocità di inferenza e robustezza balistica.
3. **OpenCV DNN Pose**: Inserimento diretto di un modulo Deep Learning via rete pre-addestrata caricata nativamente tramite il modulo Caffe/DNN di OpenCV.
4. **OpenPose Wrapper**: Simulazione deterministica e computazionalmente accurata dell'approccio *Bottom-Up* basato su *Part Affinity Fields (PAF)*, utile per mappare il carico computazionale elevato tipico di questa architettura in assenza di un SDK con accelerazione GPU dedicata.

---

## 📈 Metriche di Valutazione Estratte

Per ciascun video inserito all'interno della pipeline, lo script calcola e valida in tempo reale:
* **Tempo di Elaborazione (Latenza e FPS istantanei)**: Misura del costo computazionale e della fattibilità dell'applicazione *Real-Time*.
* **Confidence Score Medio**: Indice di certezza geometrica dei modelli nell'identificazione dei segmenti corporei.
* **Copertura delle Regioni Corporee (Keypoint Coverage)**: Percentuale di landmark validati con successo, suddivisi per macro-aree (Arti superiori, inferiori, tronco e viso) per mappare l'aderenza ai movimenti armonici del Tai Chi.

---

## 📁 Struttura del Progetto

La struttura delle cartelle deve essere configurata come segue per consentire l'auto-scansione automatica da parte dello script:
├── benchmark_hpe.py          # Script Python principale di benchmark
├── requirements.txt          # Elenco delle dipendenze del progetto
├── README.md                 # Questo file di documentazione
└── dataset/                  # Cartella principale del dataset video (.mp4)
    ├── Posa_1_Difficile/
    │   └── Posa_1_Difficile.mp4
    ├── Posa_2_Facile/
    │   └── Posa_2_Facile.mp4
    └── Posa_7_Facile/
        └── Posa_7_Facile.mp4