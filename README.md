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

---

### 🧩 Explanation of Key Folders

| Folder | Purpose |
|--------|----------|
| **inference_image/** | Store your input applicant images for verification |
| **face_database/** *(auto-created)* | Stores cropped faces assigned unique IDs |
| **embeddings/** *(auto-created)* | Contains LangChain FAISS vectorstore for duplicate checks |
| **weights/** | Place downloaded RetinaFace pretrained weights here |
| **logs/** *(auto-created)* | Stores execution logs with timestamps |

---

### ⚠️ Other Folders

Other supporting folders such as `models`, `utils`, `detection`, and `widerface_evaluate`  
will be automatically created when you clone the RetinaFace repository by **biubug6**.

---

## ✅ What You Need to Add Manually

| Required File | Location | Source |
|----------------|-----------|---------|
| `Resnet50_Final.pth` | `weights/` | Download from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) |
| `requirements.txt` | Root directory | List of dependencies |
| `README.md` | Root directory | Documentation (this file) |

---

## ⚙️ Next Steps

Once your folder structure looks like the tree above:
1. Activate your environment and install dependencies  
   ```bash
   pip install -r requirements.txt



<img width="720" height="500" alt="image" src="https://github.com/user-attachments/assets/be97deca-ffdb-468a-9416-02642050837c" />


