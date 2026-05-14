from langchain_core.documents import Document

doc = Document(
    page_content="this is main text content used to create the RAG",
    metadata={
        "source":"example.txt",
        "pages":1,
        "author":"Siva Kavindra",
        "date_created":"2025-01-01"
    }
)



import os
os.makedirs("data", exist_ok=True)
os.makedirs("data/text_files", exist_ok=True)


sample_text = {
    "data/text_files/neural_network.txt" : """ Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns and enable tasks such as pattern recognition and decision-making.

Neural networks are capable of learning and identifying patterns directly from data without pre-defined rules. These networks are built from several key components:
	• Neurons: The basic units that receive inputs, each neuron is governed by a threshold and an activation function.
	• Connections: Links between neurons that carry information, regulated by weights and biases.
	• Weights and Biases: These parameters determine the strength and influence of connections.
	• Propagation Functions: Mechanisms that help process and transfer data across layers of neurons.
	• Learning Rule: The method that adjusts weights and biases over time to improve accuracy.
	
	1. Input Computation: Data is fed into the network.
	2. Output Generation: Based on the current parameters, the network generates an output.
	3. Iterative Refinement: The network refines its output by adjusting weights and biases, gradually improving its performance on diverse tasks.
In an adaptive learning environment:
	• The neural network is exposed to a simulated scenario or dataset.
	• Parameters such as weights and biases are updated in response to new data or conditions.
	• With each adjustment, the network’s response evolves allowing it to adapt effectively to different tasks or environments.

Layers in Neural Network Architecture
Layers in Neural Network Architecture
	1. Input Layer: This is where the network receives its input data. Each input neuron in the layer corresponds to a feature in the input data.
	2. Hidden Layers: These layers perform most of the computational heavy lifting. A neural network can have one or multiple hidden layers. Each layer consists of units (neurons) that transform the inputs into something that the output layer can use.
	3. Output Layer: The final layer produces the output of the model. The format of these outputs varies depending on the specific task like classification, regression.
Working of Neural Networks
1. Forward Propagation
When data is input into the network, it passes through the network in the forward direction, from the input layer through the hidden layers to the output layer. This process is known as forward propagation. Here’s what happens during this phase:
1. Linear Transformation: Each neuron in a layer receives inputs which are multiplied by the weights associated with the connections. These products are summed together and a bias is added to the sum. This can be represented mathematically as:
	z=w_1 x_1+w_2 x_2+…+w_n x_n+b 
where
	• w represents the weights
	• x represents the inputs
	• b is the bias
2. Activation: The result of the linear transformation (denoted as z) is then passed through an activation function. The activation function is crucial because it introduces non-linearity into the system, enabling the network to learn more complex patterns. Popular activation functions include ReLU, sigmoid and tanh.
2. Backpropagation
After forward propagation, the network evaluates its performance using a loss function which measures the difference between the actual output and the predicted output. The goal of training is to minimize this loss. This is where backpropagation comes into play:
	• Loss Calculation: The network calculates the loss which provides a measure of error in the predictions. The loss function could vary; common choices are mean squared error for regression tasks or cross-entropy loss for classification.
	• Gradient Calculation: The network computes the gradients of the loss function with respect to each weight and bias in the network. This involves applying the chain rule of calculus to find out how much each part of the output error can be attributed to each weight and bias.
	• Weight Update: Once the gradients are calculated, the weights and biases are updated using an optimization algorithm like stochastic gradient descent (SGD). The weights are adjusted in the opposite direction of the gradient to minimize the loss. The size of the step taken in each update is determined by the learning rate.
3. Iteration
This process of forward propagation, loss calculation, backpropagation and weight update is repeated for many iterations over the dataset. Over time, this iterative process reduces the loss and the network's predictions become more accurate.
Through these steps, neural networks can adapt their parameters to better approximate the relationships in the data, thereby improving their performance on tasks such as classification, regression or any other predictive modeling.
Example of Email Classification
Let's consider a record of an email dataset:
Email ID	Email Content	Sender	Subject Line	Label
1	"Get free gift cards now!"	spam@example.com	"Exclusive Offer"	1
To classify this email, we will create a feature vector based on the analysis of keywords such as "free" "win" and "offer"
The feature vector of the record can be presented as:
	• "free": Present (1)
	• "win": Absent (0)
	• "offer": Present (1)
How Neurons Process Data in a Neural Network
In a neural network, input data is passed through multiple layers, including one or more hidden layers. Each neuron in these hidden layers performs several operations, transforming the input into a usable output.
1. Input Layer: The input layer contains 3 nodes that indicates the presence of each keyword.
2. Hidden Layer: The input vector is passed through the hidden layer. Each neuron in the hidden layer performs two primary operations: a weighted sum followed by an activation function.
Weights:
	• Neuron H1: [0.5,−0.2,0.3]
	• Neuron H2: [0.4,0.1,−0.5]
Input Vector: [1,0,1]
Weighted Sum Calculation
	• For H1: (1×0.5)+(0×−0.2)+(1×0.3)=0.5+0+0.3=0.8
	• For H2: (1×0.4)+(0×0.1)+(1×−0.5)=0.4+0−0.5=−0.1
Activation Function
Here we will use ReLu activation function:
	• H1 Output: ReLU(0.8)= 0.8
	• H2 Output: ReLu(-0.1) = 0
 """
}

for filepath,content in sample_text.items():
    with open(filepath,"w",encoding="utf-8") as f:
        f.write(content)

print("Sample text files is created")

## read the text created in txt file using text loader

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader

loader = TextLoader("data/text_files/neural_network.txt",encoding="utf-8")
document = loader.load()

## this is the document structure
print(document)


## directory load
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader


## encoding is used to convert the text to bytes
## used only in .txt,.json,.csv file
## not used in pdf file because it may contain text,images,fonts etc
dir_loader = DirectoryLoader(
    "data/pdf_files",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)

document = dir_loader.load()
print(document)

print("PDF file is loaded")


## RAG Pipeline

import os
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def process_all_pdf(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("*.pdf"))

    print("Length of PDF files:",len(pdf_files))

    for pdf_file in pdf_files:
        print("Processing",pdf_file,"...")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            document = loader.load()

            for doc in document:
                doc.metadata['source_file']=pdf_file.name
                doc.metadata['file_type']='pdf'

            all_documents.extend(document)
            print("Loaded all the pdf")
        except Exception as e:
            print(e)


    print("len(all_documents)",len(all_documents))
    return all_documents

all_pdf = process_all_pdf("data/pdf_files")


## splitting the data into chunks


def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )

    split = text_splitter.split_documents(documents)
    print("len(split_docs)",len(split),"into chunks" )

    if split:
        print("Split documents:")
        print(f"Content : {split[0].page_content[:200]} documents")
        print(f"Metadata {split[0].metadata}")

    return split

split_docs = split_documents(all_pdf)


class EmbeddingManger:
    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None



import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,Dict,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print("Loaded embedding model")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded")
        except Exception as e:
            print(e)

    def generate_embedding(self,texts:List[str]) ->np.ndarray:
        print("Generating embedding")
        embedding = self.model.encode(texts)
        print("Embedding shape",embedding.shape)
        return embedding

embedding_manager = EmbeddingManager()


class VectorStore:
    def __init__(self ,collection_name:str="pdf_documents", persist_directory:str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialise_store()

    def _initialise_store(self):
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings"}
            )

            print(f"Vector store initialised Collection {self.collection_name}")
            print(f"Existing Document in collection {self.collection.count()}")
        except Exception as e:
            print(e)

    def add_documents(self,document:List[Any],embeddings:np.ndarray):

        if len(document)!=len(embeddings):
            raise ValueError("The number of documents does not match the number of embeddings")

        print("Adding documents")

        ids = []
        metadatas = []
        document_text=[]
        embeddings_list=[]

        for i,(doc,embedding) in enumerate(zip(document,embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"

            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"]=i
            metadata["content_length"]=len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print("Loaded existing collection")
            except:
                # If not exists, create new one
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "PDF document embeddings"}
                )
                print("Created new collection")

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=document_text,
            )

            print(f"Successfully added {len(document)} documents to vector store")

            print(f"Total documents added to vector store: {self.collection.count()}")

        except Exception as e:
            print(e)
            raise

vector_store = VectorStore()

texts = [doc.page_content for doc in split_docs]
print(texts)
embeddings = embedding_manager.generate_embedding(texts)

vector_store.add_documents(split_docs,embeddings)

