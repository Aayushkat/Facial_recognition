import chromadb
from deepface import DeepFace

# configuring
# MUST match the exact folder name from ingest.py
DB_PATH = "./my_vector_db" 
# MUST match the exact collection name from ingest.py
COLLECTION_NAME = "Faces"   
MODEL_NAME = "Facenet512"   #Here the variable MODEL_NAME is storirng the name of the model used to encode the faces

print("Connecting to database...")
client = chromadb.PersistentClient(path=DB_PATH)

# We use get_collection (not create) because we only want to READ now
try:   #here we are checking the whether the collection(vector table) exist if it does then it is stored the the variable *collection*
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully loaded collection '{COLLECTION_NAME}' with {collection.count()} faces.")# count - returns the number of records in the collection.
except Exception as e:
    print(f"Error loading collection: {e}")
    print("The Ingest script didnt functioned properly ")
    exit()

def identify_face(img_path):
    print(f"Scanning {img_path}...")
    try:
        # 1. Generate vector for the unknown face
        target_embedding_objs = DeepFace.represent(#target_embedding stores the list of dictionaries
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True #here this paramter is true,therefore this will search the faces in the test image and if unable to find any face it return the valie error exception
        )
        # DeepFace returns a list (in case of multiple faces), we take the first element which is a facial embeeidng dictionary
        target_embedding = target_embedding_objs[0]["embedding"]

        # 2. Query ChromaDB for the 1 nearest neighbor
        results = collection.query(
            query_embeddings=[target_embedding],
            n_results=1 #THis parameter determines how many results should query return 
        )
        
        # 3. Check if we got any result at all
        if not results["ids"] or not results["ids"][0]:
            return "Unknown", 1.0

        # 4. Extract data
        match_name = results["metadatas"][0][0]["name"]
        distance = results["distances"][0][0]

        # 5. Thresholding (Cosine Distance)
        # 0.0 = Perfect Match
        # 0.4 = Good Match
        # > 0.5 = Probably different person
        if distance < 0.5:
            return match_name, distance
        else:
            return "Unknown (Low Similarity)", distance

    except ValueError:
        return "No face detected in image", 0.0
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# --- TEST ---
# Make sure you have a file named 'test.jpg' in your folder!
name, score = identify_face("test.jpg")

print(f"\n RESULT: {name}")
print(f"   Distance Score: {score:.4f}")