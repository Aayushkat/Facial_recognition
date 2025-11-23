import os
import chromadb #it is the vector database
from deepface import DeepFace #it is a deep neural network trained by meta to help face
#init a chromadab (persistent client )
#this creates a folder with name my_vector_db to store the data
#           ------configuring the database------

chroma_client=chromadb.PersistentClient(path="my_vector_db")


#Persistent clinet init the DB in the disk storage rather than the RAM
#Ephermeral client init the DB in the RAM where it can be lost after the session end 

#embedding here is the vectors of any object like pictures , voice , text etc  

#Collections are where you'll store your embeddings, documents, and any additional
#metadata. Collections index your embeddings and documents,
#and enable efficient retrieval and filtering.
collection = chroma_client.get_or_create_collection(name="Faces",metadata={"hnsw:space":"cosine"})
'''                                                                             
HNSW:Hierarchical navigable small world
HNSW stores similar vectors to each other , like in each others vicinity.Basically making the cluster of similar vectors
and it prevent us brute forcing the vectors. 

_It is a ANN (approximate nearest neighbour) , ANN use cluster, key hashing ,quatization methods to store the vectors  

_Here we are passing the metadata . it is not necassry to mention the metadata in the
arguement of the create collection

_and the cosine is the *distace metric* here , there are two type also another metric like euclidean distance called L2


_if HSNW is the engineer to build the houses in the neighbourhod then the cosine and the euclidean distance are the distace measurinng unit
of how the houses will be build . Now the houses here are the vectors , the engineer here is the HSNNW

'''
 


DB_path="Face_db"                                           #used for managing the file directory
Model_name="Facenet512"                                     #name of the model in the deepface to identify faces

for filename in os.listdir(DB_path):                        #it goes thru all the directory named in the arguement and storing a string variable named 'filename'
    if filename.lower().endswith(('.png',".jpg",".jpeg")):  #.lower() makes the characeres in the string all to lowercase
                                                            # .endswith() will check all the suffix mentioned in the arguement and return the true or false 
        file_path=os.path.join(DB_path,filename)            #this statement makes a variable of the  which containns the fffull usable path to acces the photo/file
        person_name=os.path.splitext(filename)[0]           #it splits the name of the file or should i say root and the extension of it eg rahul.png it split .png and rahul\
        try:
            #Now we generate the embedding
            #output will be a list of dictionary
            #we will use the first face (dictionary) in the image
            embedding_obj=DeepFace.represent(img_path=file_path,
                                             model_name=Model_name,
                                             enforce_detection=False)#enforce detection is basically to handle cases of 
                                                                    #where face is detected or not 
                                                                    #if it is *True* and then the face is not detected it throws a value exception error and it stops the execution of the program
                                                                    #if the parameter *False* then if the face is not detected then the it skip in findin the face and then it makes the the vector embedding of the whole image

            


            #extract the vector from the output which will be in various dictioanary
            embedding=embedding_obj[0]["embedding"]
            '''Example of the output
            [                                                  //the dictionaries are in the list 
                {
                    'embedding': [0.123, 0.456, ..., 0.789],  // The facial embedding vector

                    'facial_area': {'x': 100, 'y': 50, 'w': 150, 'h': 200},  // A dictionary containing the bounding box coordinates of the detected
                                                                            // face within the image, usually in the format 
                                                                             //{'x': int, 'y': int, 'w': int, 'h': int}.
                    'detector_backend': 'mtcnn',
                    'model': 'VGG-Face'
                },
                # ... potentially more dictionaries if multiple faces are detected
            ]'''

            #store it in the chromaDB
            collection.add(
                ids=[filename],                               #unique ID for each vector
                embeddings=[embedding],                        #vector that will be stored
                metadatas=[{"name":person_name}]                                   #it stores th vector embedding with the anme of the person in diffreent column
            )
            print(f"Added: {person_name}")                      #it is to show the user that the person name is stored
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"Finished! Database has {collection.count()} faces.")       