<div align="center">

<!-- Reliable Header Banner (repo-hosted) -->
<img src="https://raw.githubusercontent.com/Daivik1520/face-recognition/main/README_banner.png" alt="Face Recognition — AI Attendance & Security" width="100%">

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=900&color=00D9FF&center=true&vCenter=true&width=700&lines=95%25+Accuracy+%7C+%3C100ms+Recognition+%7C+30+FPS+Live+Detection;FastAPI+%2B+InsightFace+%2B+Next.js+%7C+Privacy-First+Local+Processing" alt="Tagline" />
</p>

<!-- Badges -->
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.104+-0FA47E?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="Next.js" src="https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white">
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white">
</p>

<p align="center">
  <a href="#-demo">🚀 Demo</a> •
  <a href="#-features">✨ Features</a> •
  <a href="#-quick-start">🛠️ Quick Start</a> •
  <a href="#-architecture">🏗️ Architecture</a> •
  <a href="#-api-documentation">📖 API</a> •
  <a href="#-deployment">🌐 Deploy</a> •
  <a href="#-security--privacy">🛡️ Security</a> •
  <a href="#-troubleshooting">🔧 Troubleshoot</a> •
  <a href="#-performance-metrics">📊 Performance</a> •
  <a href="#-roadmap">🗺️ Roadmap</a> •
  <a href="#-contributing">🤝 Contributing</a>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

</div>

## 🎯 Why This Project

A production-grade, end-to-end face recognition platform for **attendance** and **access control**. It delivers:
- Real-time recognition from live camera streams
- Accurate embeddings with **InsightFace (ArcFace)** and **FAISS** vector search
- A modern **Next.js** dashboard for enrollment, live feed, analytics
- **Local-first privacy** with optional cloud deployment

---

## 🌟 Features

- 🎥 30 FPS live recognition with **< 100ms** response
- 👥 Guided multi-angle **enrollment** workflow
- 🧠 InsightFace model pack (detector, recognizer, landmarks)
- 🔎 FAISS similarity search with tunable threshold
- 📊 Analytics: daily/weekly trends, CSV/JSON export
- 🗃️ Per-person history, confidence scoring
- 🔒 Local storage; GDPR-aligned practices
- 📱 Responsive UI (desktop-first)

---

## 🚀 Demo & Requirements

| Component | Requirement |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| RAM | 8GB+ (recommended) |
| Webcam | HD camera |

> Tip: For GPU, install CUDA-enabled ONNX/OpenCV where available.

---

## 🛠️ Quick Start

<details open>
<summary><b>Automated setup (recommended)</b></summary>

```bash
# 1) Clone
git clone https://github.com/Daivik1520/face-recognition.git
cd face-recognition

# 2) Install & fetch models
chmod +x scripts/setup.sh
./scripts/setup.sh        # Windows: scripts\setup.bat

# 3) Run both services via helper
python3 run_dev.py        # Frontend: http://localhost:3000  Backend: http://localhost:8000
```

</details>

<details>
<summary><b>Manual setup</b></summary>

```bash
# Backend
cd backend && poetry install && cd ..

# Frontend
cd frontend && npm install && cd ..

# Models
cd scripts && python3 download_models.py && cd ..

# Launch (concurrently)
python3 run_dev.py
```

</details>

---

## 🏗️ Architecture

```
📷 Camera → 🎯 Detector → 🧠 Embeddings → 🔍 FAISS → ✅ Identity → 🗂️ Log
                      ↘︎                           ↘︎
                     UI                            Analytics
```

- Frontend: **Next.js App Router** (Live, Enrollment, Analytics, Attendance)
- Backend: **FastAPI** (Pydantic v2) — recognition, enroll, analytics routes
- Models: **InsightFace Buffalo_L** (det_10g, w600k_r50, 1k3d68, genderage)
- Storage: `./data/models`, `./data/processed` (embeddings/attendance)

### Project Structure
```
face-recognition/
├── backend/
│   └── src/app/ (routes, schemas, deps)
│       ├── face_system.py
│       ├── analytics.py
│       ├── augmentation.py
│       └── models/
├── frontend/
│   └── src/app/ (dashboard, live, analytics, attendance)
├── scripts/ (setup.sh, setup.bat, download_models.py)
├── data/ (models, processed, augmented)
└── run_dev.py
```

---

## 📖 API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Core Endpoints
```http
POST /api/recognize
# multipart/form-data: file=@image
→ { recognized, person, confidence, attendance_logged, timestamp }

POST /api/enroll
# json: { name, images:[base64...], department }
→ { success, person_id, embeddings_count, accuracy_score }

GET /api/analytics/dashboard
→ { total_enrolled, today_attendance, recognition_accuracy, trends }
```

---

## 🌐 Deployment

<details>
<summary><b>Docker Compose</b></summary>

```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

</details>

<details>
<summary><b>Vercel + Server</b></summary>

- Deploy frontend to **Vercel** (set `NEXT_PUBLIC_API_URL`)
- Deploy backend to a VM/container; mount `./data`

</details>

---

## 🛡️ Security & Privacy

- Local processing; no face images sent to third parties
- Optional encryption at rest for embeddings
- Configurable retention; anonymized analytics possible
- CORS, input validation, strict file-type checks

---

## 🔧 Troubleshooting

- Camera not found → set `CAMERA_INDEX=0..n`; check permissions
- Low accuracy → lower `SIMILARITY_THRESHOLD` (0.65→0.6); enroll more angles
- Low FPS → reduce resolution; increase `FRAME_SKIP`; enable GPU
- Ports busy → change `FRONTEND_PORT` / `API_PORT`; kill PIDs using 3000/8000

---

## 📊 Performance Metrics

| Metric | Target | Notes |
|---|---|---|
| Recognition latency | < 100ms | single face, HD webcam |
| Throughput | 30 FPS | per stream |
| Accuracy | 94–96% | good lighting & angles |

---

## 🗺️ Roadmap

- Multi-camera grids & ingestion
- Liveness detection & anti-spoofing
- Role-based auth, SSO, audit logs
- PostgreSQL/Redis options, backups
- Cloud/edge hybrid sync

---

## 🤝 Contributing

- Fork → feature branch → PR with screenshots/logs
- Python (PEP 8) & TypeScript best practices
- Add tests where practical; keep functions focused

---

## 📄 License

MIT — see `LICENSE`.
