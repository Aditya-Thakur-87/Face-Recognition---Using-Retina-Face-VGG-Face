# 🔍 Face Authentication Pipeline using RetinaFace + VGGFace2 + FAISS
### *An AI System for Duplicate Applicant Detection — IndiaAI Face Authentication Challenge 2025*

---

## 🧠 Overview

This repository implements a **complete face authentication system** that:
- Detects faces in applicant photos using **RetinaFace**  
- Aligns and crops detected faces using an **MTCNN-based aligner**
- Generates **VGGFace2 embeddings** (via `facenet-pytorch`)
- Stores and compares embeddings using **LangChain + FAISS**
- Automatically classifies new applicants as **Duplicate**, **Review**, or **Unique**

This prevents multiple submissions under different identities and ensures **fair selection** in government or exam verification systems.

---

## 🏗️ System Workflow

| Step | Module | Description |
|------|---------|-------------|
| **1. Face Detection** | `RetinaFace (biubug6)` | Detects all visible faces in the input image |
| **2. Face Alignment** | `face_alignment.py` | Normalizes face orientation, eyes alignment |
| **3. Embedding Generation** | `InceptionResnetV1 (VGGFace2)` | Converts each aligned face into a 512-D feature vector |
| **4. Similarity Search** | `LangChain + FAISS` | Computes cosine similarity with database embeddings |
| **5. Decision** | `face_pipeline.py` | Assigns unique ID if new, else returns existing duplicate ID |
| **6. Logging** | `face_database/`, `embeddings/` | Saves aligned faces, scores, and FAISS index |

---

## 📁 Project Structure
<img width="720" height="500" alt="image" src="https://github.com/user-attachments/assets/fae1cc4c-2554-4edb-aad1-27d178e9767a" />


