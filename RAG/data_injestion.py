from langchain_core.documents import Document


"""
This is the structure of the Document that is used for the chunking purpose
"""
doc = Document(
    page_content="this is main text content used to create the RAG",
    metadata={
        "source" : "example.txt",
        "page":1,
        "author":"Siva Kavindra",
        "date_created":"2026-04-19"
    }
)

import os
os.makedirs("data/text_files",exist_ok=True)
sample_text = {
    "data/text_files/neural_network.txt":"""Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns and enable tasks such as pattern recognition and decision-making.Neural networks are capable of learning and identifying patterns directly from data without pre-defined rules. These networks are built from several key components:
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
""" ,
    "data/text_files/genAI.txt" : """GEN AI - Generating new content

Initially simple table is easy to handle and we have structured data so easy.
If need to find the given image is cat or dog here data are complicated one so comes deep learning  and neural network


Statisctical Method -> neural netwrok -> recural nural network -> generating remaing words. (Langiage Model)->Transformers

Example Anjela,Thanks for 
Contacring me
Reaching out
Finoshing project
Each have differnet probabilty to generate the next words.

We give the datasets or sample to neural network here it does self supervised learning.
Neural netwrok will predict the next words
So many layers in neural network is LLM
Parameters means weights.

Transformer
Google - BERT (text model)
Similarly there will be for image models.
Text - Text Model
BERT
GPT
Text to image
DALL - E
Stable Diffusion
Text to Video
OpenAI Sora

Statistical Probabilty and randomness

GPT - Chatgpt
PaLM 2 - Google
LLaMA- Meta

Open Ai used RLHF to make less toxic
LLM works on database alone that is given."""
}

for filename,content in sample_text.items():
    with open(filename,"w",encoding="utf-8") as f:
        f.write(content)





