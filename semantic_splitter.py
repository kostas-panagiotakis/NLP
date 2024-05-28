import re
!pip install numpy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# !pip install -U sentence-transformers

def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences
    
    
def SemanticEmbedder(filepath):
 
    # 1) Open file
    with open(filepath) as file:
        essay = file.read()
 
    # 2) Splitting file into sentences chunks + turn them into a  list of dictionaries
    single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]

    # 3) Combine 3 adjacent sentences togheter
    sentences = combine_sentences(sentences)
    sentences_list_combined = [x['combined_sentence'] for x in sentences]	 
 
    # 4) Embed the combined sentences with SBERT
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.zeros((len(sentences_list_combined), model.get_sentence_embedding_dimension()))
    for i, sentence in enumerate(sentences_list_combined):
        embeddings[i] = model.encode(sentence).tolist()

    # 5) Add the embedded combined sentence list to the dictionary
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    # 6) Compute cosine distances between sequential embedding pairs i, i+1
    distances, sentences = calculate_cosine_distances(sentences)
 
    # Split the text based on semantic, grouping sentences into chunks
    start_index = 0
    chunks = []
    breakpoint_percentile_threshold = 95
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) 
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
	    
	      # Update the start index for the next group
        start_index = index + 1

	  # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
 
    return chunks
 
 
 
 
 
 
 
 
 
 
