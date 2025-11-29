# Facial recognition with Vector databases

This is a high-speed facial recognition and verification system that converts faces into unique numerical vectors (embeddings) and uses a specialized database for instant searching.

---

### Key Technologies Used:
* **Python 3.11.0**: Main language
* **DeepFace 0.0.95**: Handles facial detection and vector encoding
* **Chromadb 1.3.5**: Stores and searches the facial embeddings
* **Tensorflow 2.20.0 and Keras 2.20.1**: Core deep learning backend for DeepFace
---

# Getting started

## Prerequisites

### **For Windows User**

1. Make sure you have downloaded the **Microsoft Visual C++ Redistributable (x64)**. This is a critical dependency for TensorFlow and Keras to run on Windows.

### Installation Steps

1. **Clone the Repository:**
    ```bash
    git clone [https://github.com/Aayushkat/Facial_recognition.git](https://github.com/Aayushkat/Facial_recognition.git)
    ```
2. **Change Directory:**
    ```bash
    cd Facial_recognition
    ```
3. **Initialize a Python Virtual Environment:**
    ```bash
    python -m venv virtual_env
    ```
4. **Activate the Environment:**
    ```bash
    # On Windows:
    virtual_env/Scripts/activate
    
    # On Linux/macOS (if applicable):
    # source virtual_env/bin/activate
    ```
5. **Install the Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Steps for Utilisation

There are two main scripts: **Ingestion** (to build the database) and **Recognition** (to search the database).

### Step 1: Ingest Faces and Build the Database (`ingest.py`)

This step scans your data folder and populates the vector database.

1.  **Add Images:** Place the training images (e.g., `Messi.jpg`, `Ronaldo.jpg`) into the `face_db/` folder.
2.  **Run Ingestion:**
    ```bash
    python ingest.py
    ```
    *Note: The first time you run this, DeepFace will automatically download large model weight files (`.h5`) into the hidden `.deepface/` folder.*

### Step 2: Recognize a Face (`recognize.py`)

This step uses the embeddings you just created to identify an unknown face.

1.  **Prepare Test Image:** Place an image you wish to test in the project root folder and name it **`test.jpg`**.
2.  **Run Recognition:**
    ```bash
    python recognize.py
    ```
    *The output will display the name of the closest match and the similarity distance score.*

