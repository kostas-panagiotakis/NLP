# Install necessary packages
!pip install -U -q "langchain" "transformers==4.31.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.21.0" "bitsandbytes==0.40.2" "trl==0.4.7" "safetensors>=0.3.1"
!pip install --upgrade langchain-community
!pip install -q -U faiss-cpu tiktoken sentence-transformers
!pip install huggingface-hub -q
!pip install jq


# Import necessary libraries
import numpy as np
import pandas as pd
from datasets import load_dataset
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import JSONLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

import json
from typing import List, Tuple
import torch
import transformers
from huggingface_hub import notebook_login
from pathlib import Path

import re  # Regular expression library
import transformers
import transformers
from sentence_transformers import SentenceTransformer  # For sentence embedding
from sklearn.metrics.pairwise import cosine_similarity  # For computing distance between sentence embeddings

# Download the text file
!wget https://github.com/mattf/joyce/blob/master/james-joyce-ulysses.txt

# Load the text file
ulysses = TextLoader(file_path='james-joyce-ulysses.txt')
ulysses_loaded = ulysses.load()

def clean_documents(documents):
    """
    Clean the document list by removing specific characters and converting to lowercase.
    """
    cleaned_documents = []

    for doc in documents:
        modified_doc = doc.copy()  # Create a copy of the document
        modified_doc.page_content = modified_doc.page_content.replace('\\r', '')  # Remove '\r' characters
        modified_doc.page_content = modified_doc.page_content.replace('\\', '')  # Remove '\\' characters
        modified_doc.page_content = modified_doc.page_content.lower()  # Convert to lowercase
        cleaned_documents.append(modified_doc)  # Add modified document to the new list

    return cleaned_documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # the character length of the chunk
    chunk_overlap = 100, # the character length of the overlap between chunks
    length_function = len, # the length function - in this case, character length (aka the python len() fn.)
)


# Transform and clean the documents
ulysses_loaded_documents = text_splitter.transform_documents(ulysses_loaded)[196:2018]
ulysses_loaded_documents = clean_documents(ulysses_loaded_documents)
ulysses_recursive = ulysses_loaded_documents


def combine_sentences(sentences, buffer_size=1):
    """
    Combine sentences with a given buffer size.
    """
    for i in range(len(sentences)):

        combined_sentence = ''
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences):
    """
    Calculate cosine distances between sentence embeddings.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance

    return distances, sentences


def SemanticEmbedder(filepath):
    """
    Embed sentences in a text file and return chunks based on cosine distance breakpoints.
    """

    with open(filepath) as file:
        essay = file.read()

    single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]

    sentences = combine_sentences(sentences)
    sentences_list_combined = [x['combined_sentence'] for x in sentences]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.zeros((len(sentences_list_combined), model.get_sentence_embedding_dimension()))
    for i, sentence in enumerate(sentences_list_combined):
        embeddings[i] = model.encode(sentence).tolist()

    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    distances, sentences = calculate_cosine_distances(sentences)
    start_index = 0
    chunks = []
    breakpoint_percentile_threshold = 87
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    for index in indices_above_thresh:

        end_index = index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index + 1
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks
    
# Process the text file
chunks = SemanticEmbedder('/content/james-joyce-ulysses.txt')

# Display chunks
for i, chunk in enumerate(chunks[10:15]):
    buffer = 2000
    print(f"Chunk #{i}")
    print(chunk[:buffer].strip())

# Convert chunks to JSON format
json_data = []
for i, chunk in enumerate(chunks[2:]):
    json_object = {
        "page_content": chunk,
        "metadata": "dubliners.txt"
    }
    json_data.append(json_object)


# Write JSON objects to a file
with open("output.json", "w") as json_file:
    for item in json_data:
        json.dump(item, json_file)
        json_file.write("\n")

file_path = '/content/output.json'

# Read JSON data
json_data = []
with open(file_path, 'r') as file:
    for line in file:
        json_data.append(json.loads(line))


loader = JSONLoader(
    file_path='/content/output.json',
    jq_schema='.',
    content_key='page_content',
    json_lines=True
)


ulysses_semantic = loader.load()

def analyze_word_frequency(n_words, df):
    """
    Analyzes the word frequency in a DataFrame containing text documents.

    Args:
    - n_words (int): Number of most frequent words to select.
    - df (DataFrame): DataFrame containing text documents.

    Returns:
    - DataFrame: DataFrame containing the specified number of most frequent words and their counts.
    """
    concatenated_text = ""
    for doc in df:
        concatenated_text += doc.page_content
    words = concatenated_text.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1])
    most_frequent_words = sorted_word_counts[-n_words:]
    most_frequent_df = pd.DataFrame(most_frequent_words, columns=['Word', 'Count'])
    most_frequent_df = most_frequent_df.sort_values(by='Count', ascending=False)
    return most_frequent_df

def plot_word_frequency(df, title):
    """
    Plots the word frequency data.

    Args:
    - df (DataFrame): DataFrame containing word frequency data.
    - title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Word', data=df, hue='Word', palette='viridis', legend=False)
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.show()

def doc_stat(df):
    """
    Calculates statistics about the documents in the DataFrame.

    Args:
    - df (DataFrame): DataFrame containing documents.

    Returns:
    - dict: A dictionary containing document statistics.
    """
    documents = 0
    characters = 0
    source_files = []
    for doc in df:
        documents += 1
        characters += len(doc.page_content)
        source_files.append(doc.metadata['source'])
    source_files = np.unique(source_files)
    return {'Documents': documents, 'Characters': characters, "Sources": source_files}


# Analyze and plot word frequency
n_words = 30
ulysses_recursive_most_frequent_df = analyze_word_frequency(n_words, ulysses_recursive)
ulysses_semantic_most_frequent_df = analyze_word_frequency(n_words, ulysses_semantic)
plot_word_frequency(ulysses_recursive_most_frequent_df, f'{n_words} Most Frequent Words in Total with Recursive')
plot_word_frequency(ulysses_semantic_most_frequent_df, f'{n_words} Most Frequent Words in Total with Semantic')

# Document statistics
ulysses_recursive_stats = doc_stat(ulysses_recursive)
ulysses_semantic_stats = doc_stat(ulysses_semantic)
doc_df = pd.DataFrame([ulysses_recursive_stats, ulysses_semantic_stats], index=['Ulysses Recursive', 'Ulysses Semantic'])
print(doc_df)
    

def get_vectors_from_store(vector_store) -> List:
    """
    Access the vectors in the FAISS vector store.
    
    Args:
    - vector_store (FAISS): The FAISS vector store.
    
    Returns:
    - List: A list of vectors from the vector store.
    """
    
    # Access the vectors in the FAISS vector store
    vectors = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
    return vectors

def create_vector_store(embed_model_id, doc, cache_dir="./cache/"):
    """
    Create a vector store using the specified embedding model and documents.

    Args:
    - embed_model_id (str): The ID of the embedding model.
    - doc (list): The list of documents.
    - cache_dir (str): The cache directory.

    Returns:
    - tuple: A tuple containing the store, core embeddings model, embedder, and vector store.
    """
    store = LocalFileStore(cache_dir)
    core_embeddings_model = HuggingFaceEmbeddings(
        model_name=embed_model_id
    )
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=embed_model_id
    )
    vector_store = FAISS.from_documents(doc, embedder)

    return store, core_embeddings_model, embedder, vector_store

store_MiniLM_recursive, core_embeddings_model_MiniLM_recursive, embedder_MiniLM_recursive, vector_store_MiniLM_recursive = \
create_vector_store('sentence-transformers/all-MiniLM-L6-v2', ulysses_recursive)
raw_vector_store_MiniLM_recursive = get_vectors_from_store(vector_store_MiniLM_recursive)

store_MiniLM_semantic, core_embeddings_model_MiniLM_semantic, embedder_MiniLM_semantic, vector_store_MiniLM_semantic = \
create_vector_store('sentence-transformers/all-MiniLM-L6-v2', ulysses_semantic)
raw_vector_store_MiniLM_semantic = get_vectors_from_store(vector_store_MiniLM_semantic)

store_paraphrase_mpnet_recursive, core_embeddings_model_paraphrase_mpnet_recursive, \
embedder_paraphrase_mpnet_recursive, vector_store_paraphrase_mpnet_recursive = \
create_vector_store('sentence-transformers/paraphrase-mpnet-base-v2', ulysses_recursive)
raw_vector_store_paraphrase_mpnet_recursive = get_vectors_from_store(vector_store_paraphrase_mpnet_recursive)

store_paraphrase_mpnet_semantic, core_embeddings_model_paraphrase_mpnet_semantic, \
embedder_paraphrase_mpnet_semantic, vector_store_paraphrase_mpnet_semantic = \
create_vector_store('sentence-transformers/paraphrase-mpnet-base-v2', ulysses_semantic)
raw_vector_store_paraphrase_mpnet_semantic = get_vectors_from_store(vector_store_paraphrase_mpnet_semantic)

store_all_mpnet_recursive, core_embeddings_model_all_mpnet_recursive,\
embedder_all_mpnet_recursive, vector_store_all_mpnet_recursive = \
create_vector_store('sentence-transformers/all-mpnet-base-v2', ulysses_recursive)
raw_vector_store_all_mpnet_recursive = get_vectors_from_store(vector_store_all_mpnet_recursive)

store_all_mpnet_semantic, core_embeddings_model_all_mpnet_semantic,\
embedder_all_mpnet_semantic, vector_store_all_mpnet_semantic = \
create_vector_store('sentence-transformers/all-mpnet-base-v2', ulysses_semantic)

def visualize_similarity_score(docs, number_of_docs):

    documents = [
        (f"Document {i+1}", docs[i][1]) for i, doc in enumerate(docs)
    ]

    document_titles = [doc[0] for doc in documents]
    similarity_scores = [doc[1] for doc in documents]

    plt.figure(figsize=(10, 6))
    plt.barh(document_titles, similarity_scores, color='skyblue')
    plt.xlabel('Similarity Score')
    plt.ylabel('Document')
    plt.title('Similarity Scores for Query "Who is Stephen Dedalus"')
    plt.gca().invert_yaxis()  # To display the highest score on top
    plt.show()

query = "Who is Stephen Dedalus"
embedding_vector = core_embeddings_model_all_mpnet_recursive.embed_query(query)
docs = vector_store_all_mpnet_recursive.similarity_search_by_vector(embedding_vector, k = 4)

for page in docs:
  print(page.page_content)

k = 4
number_of_docs = [i for i in range(k)]
query = "Who is Stephen Dedalus"
embedding_vector = core_embeddings_model_all_mpnet_recursive.embed_query(query)
docs = vector_store_all_mpnet_recursive.similarity_search_with_score_by_vector(embedding_vector, k = k)

for page in docs:
  print(page)

visualize_similarity_score(docs, number_of_docs)

import numpy as np

def cosine_similarity(query, vectors):

    query_norm = query / np.linalg.norm(query)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    similarities = np.dot(vectors_norm, query_norm)

    return similarities

similarities = cosine_similarity(embedding_vector, raw_vector_store_all_mpnet_recursive)
sorted_indices = np.argsort(-similarities)
sorted_similarities = similarities[sorted_indices]

sorted_vectors = [joyce_books_recursive[sorted_indices[0]].page_content,\
                  joyce_books_recursive[sorted_indices[1]].page_content,\
                  joyce_books_recursive[sorted_indices[2]].page_content,\
                  joyce_books_recursive[sorted_indices[3]].page_content]

print(sorted_similarities[0], sorted_vectors[0])
print(sorted_similarities[1], sorted_vectors[1])
print(sorted_similarities[2], sorted_vectors[2])
print(sorted_similarities[3], sorted_vectors[3])


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

distances = np.array([euclidean_distance(embedding_vector, doc_vector) for doc_vector in raw_vector_store_all_mpnet_recursive])
sorted_indices = np.argsort(distances)
sorted_distances = distances[sorted_indices]
sorted_vectors = [joyce_books_recursive[sorted_indices[0]].page_content,\
                  joyce_books_recursive[sorted_indices[1]].page_content,\
                  joyce_books_recursive[sorted_indices[2]].page_content,\
                  joyce_books_recursive[sorted_indices[3]].page_content]

print(sorted_distances[0], sorted_vectors[0])
print(sorted_distances[1], sorted_vectors[1])
print(sorted_distances[2], sorted_vectors[2])
print(sorted_distances[3], sorted_vectors[3])


def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

distances = np.array([manhattan_distance(embedding_vector, doc_vector) for doc_vector in raw_vector_store_all_mpnet_recursive])
sorted_indices = np.argsort(distances)
sorted_distances_manh = distances[sorted_indices]

sorted_vectors = [joyce_books_recursive[sorted_indices[0]].page_content,\
                  joyce_books_recursive[sorted_indices[1]].page_content,\
                  joyce_books_recursive[sorted_indices[2]].page_content,\
                  joyce_books_recursive[sorted_indices[3]].page_content]

print(sorted_distances_manh[0], sorted_vectors[0])
print(sorted_distances_manh[1], sorted_vectors[1])
print(sorted_distances_manh[2], sorted_vectors[2])
print(sorted_distances_manh[3], sorted_vectors[3])


### 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
notebook_login()

# Define the model ID
model_id = "meta-llama/Llama-2-13b-chat-hf"

# Configure BitsAndBytes quantization parameters
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,                      # Load model weights in 4-bit format
    bnb_4bit_quant_type='nf4',              # Set 4-bit quantization type to 'nf4' (Nearest Floating)
    bnb_4bit_use_double_quant=True,         # Use double quantization
    bnb_4bit_compute_dtype=torch.bfloat16   # Set the compute data type to bfloat16
)

# Load model configuration from pretrained model
model_config = transformers.AutoConfig.from_pretrained(model_id)

# Initialize the model with pretrained weights and configurations
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,                               # Load the model from the specified ID
    trust_remote_code=True,                 # Trust the remote code source
    config=model_config,                    # Use the loaded model configuration
    quantization_config=bnb_config,         # Apply the specified quantization configuration
    device_map='auto'                       # Automatically select the device for model computation
)

# Set the model to evaluation mode
model.eval()


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id
)

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=False,
    temperature=0.3,
    max_new_tokens=1000
)

# Template function for question-answering with sources
def query_and_answer(core_embeddings_model, vector_store):

    print("Hello, welcome to the Joyce Companion, here you can embark on journey of the exploration of the Joycean literature world. How may I assist you today?\n")

    while True:
        query_text = input("Please enter your question: \n")

        # Retrieve N-best documents relevant to the query
        embedding_vector = core_embeddings_model.embed_query(query_text)
        docs = vector_store.similarity_search_by_vector(embedding_vector, k=4)

        # Generate answer based on the query and retrieved documents
        result = qa_with_sources_chain({"query": query_text})

        print(result['result'])
        print("/n")

        explore_more = input("Is there anything else you would like to explore? (Yes/No): \n").strip().lower()
        if explore_more == 'no' or explore_more.startswith('n'):
            print("Thank you for using the Joyce Companion. See you next time!\n")
            break

def model_evaluation(question):

    # Encode the input question using the model's tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer.encode(question, return_tensors="pt")

    # Generate a response
    output = model.generate(inputs, max_length=1000, num_return_sequences=1, do_sample=True, temperature=0.3)

    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Question 1
question = "What kind of satirical actions does Buck Mulligan do in the first chapter of Ulysses?"
# Get the response
response = model_evaluation(question)
# Print the response
print("Response:", response)


# Question 2
question = "Can you compare Joyce's Ulysses to Homer's Odissey?"
# Get the response
response = model_evaluation(question)
# Print the response
print("Response:", response)

# Question 3
question = "Explain what green bile reference to in the first chapter of Ulysses?"
# Get the response
response = model_evaluation(question)
# Print the response
print("Response:", response)

# Question 4
question = "What does Leopold Bloom get from the pork shop in the forth chapter Calypso?"
# Get the response
response = model_evaluation(question)
# Print the response
print("Response:", response)

# Question 5
question = "Who writes a letter to Mrs. Marion Bloom in chapter four of Ulysses?"
# Get the response
response = model_evaluation(question)
# Print the response
print("Response:", response)

llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vector_store_MiniLM_recursive.as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

query_and_answer(core_embeddings_model_MiniLM_recursive, vector_store_MiniLM_recursive)

llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vector_store_paraphrase_mpnet_recursive.as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

query_and_answer(core_embeddings_model_paraphrase_mpnet_recursive, vector_store_paraphrase_mpnet_recursive)



llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vector_store_all_mpnet_recursive.as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

query_and_answer(core_embeddings_model_all_mpnet_recursive, vector_store_all_mpnet_recursive)




llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vector_store_MiniLM_semantic.as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

query_and_answer(core_embeddings_model_MiniLM_semantic, vector_store_MiniLM_semantic)



llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vector_store_paraphrase_mpnet_semantic.as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

query_and_answer(core_embeddings_model_paraphrase_mpnet_semantic, vector_store_paraphrase_mpnet_semantic)



llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vector_store_all_mpnet_semantic.as_retriever()

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)

query_and_answer(core_embeddings_model_all_mpnet_semantic, vector_store_all_mpnet_semantic)


