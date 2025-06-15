import argparse
from gensim.models import KeyedVectors
import numpy as np
import re
import sentencepiece
import torch
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
#BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer

from sentence_transformers import SentenceTransformer # For LaBSE embeddings

#from sklearn.metrics.pairwise import cosine_similarity


### MODIFY PATH ###
PRETRAINING_PATH = '../pretraining_test/modelling/output-hsb' #first_attempt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Sentence file (format: <ID>\t<sentence>)=')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file path')
    parser.add_argument('-m', '--model_name', type=str, required=True, choices=['xlmr', 'glot500', 'pretrained', 'labse', 'laser'], help='Embedding model')

    return parser.parse_args()



# Embedding manager from SimAlign
class EmbeddingLoader(object):
	def __init__(self, model: str="bert-base-multilingual-cased", device=torch.device('cpu'), layer: int=8):
		TR_Models = {
			'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer)
		}

		self.model = model
		self.device = device
		self.layer = layer
		self.emb_model = None
		self.tokenizer = None

		if model in TR_Models:
			model_class, tokenizer_class = TR_Models[model]
			self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
			self.emb_model.eval()
			self.emb_model.to(self.device)
			self.tokenizer = tokenizer_class.from_pretrained(model)
		else:
			# try to load model with auto-classes
			config = AutoConfig.from_pretrained(model, output_hidden_states=True)
			self.emb_model = AutoModel.from_pretrained(model, config=config)
			self.emb_model.eval()
			self.emb_model.to(self.device)
			self.tokenizer = AutoTokenizer.from_pretrained(model)
		#LOG.info("Initialized the EmbeddingLoader with model: {}".format(self.model))

	def get_embed_list(self, sent_batch) -> torch.Tensor: #sent_batch: List[List[str]]
		if self.emb_model is not None:
			with torch.no_grad():
				if not isinstance(sent_batch[0], str):
					inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
				else:
					inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True, return_tensors="pt")
				hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
				if self.layer >= len(hidden):
					raise ValueError(f"Specified to take embeddings from layer {self.layer}, but model has only {len(hidden)} layers.")
				outputs = hidden[self.layer]
				return outputs[:, 1:-1, :]
		else:
			return None


# Utility functions
def flatten_2D(list_of_list):
    '''Flatten a 2D list (list of list).'''
    return [element for element_list in list_of_list for element in element_list]

def delete_value_from_vector(vector, value):
    '''Delete a given value from a vector.

    To be used only when the value is in the vector.
    '''
    if value in vector:
        vector.remove(value)
        return vector
    else:
        raise ValueError('The asked value is not in the vector.')

def text_to_line(raw_text):
    r'''Split a raw text into a list of sentences (string) according to '\n'.'''
    split_text = re.split('\n', raw_text)
    if '' in split_text: # To remove empty lines
        return delete_value_from_vector(split_text, '')
    else:
        return split_text


# Processing embeddings
def get_embedding(sentence, xlmr_embeddings):
    '''Getting an embedding from XLMR.'''
    embedding_size = 768
    s_embedding = xlmr_embeddings.get_embed_list([[sentence]])
    assert s_embedding.size(dim=0) == 1, f'First dimension is not 1: {s_embedding.size()}.'
    if s_embedding.size(dim=1) > 1: # More than 1 dimension of the second dimension
        np_embedding = s_embedding.cpu().detach().numpy()[0].mean(axis=0)
        #print(np_embedding.shape)
    elif s_embedding.size(dim=1) == 1: # Only one dimension
        np_embedding = s_embedding.cpu().detach().numpy()[0][0]
    else: # Continue
        return None
    ls_embedding = np_embedding.tolist()#[0:embedding_size]
    assert len(ls_embedding) == embedding_size, f'The embedding size is different {len(ls_embedding)}'
    str_embedding = [f'{embed_value:.6f}' for embed_value in ls_embedding] # Convert format
    return str_embedding

def to_xlmr_sentence_embeddings(path, sentence_list, start_i=0):
    '''Save the embeddings from XLMR in a txt file (same format as fastText) in a batch manner.'''
    model_name_def = 'pretrained'
    if model_name_def == 'xlmr':
        model_name = 'xlm-roberta-base'
    elif model_name_def == 'glot500':
        model_name = 'cis-lmu/glot500-base' #'xlm-roberta-base'
    elif model_name_def == 'pretrained':
        model_name = 'yihongLiu/furina'
    xlmr_embeddings = EmbeddingLoader(model_name, torch.device('cpu'), layer=8)
    embedding_size = 768 #300
    
    # Initial step
    if start_i == 0:
        sentence = sentence_list[0]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        str_embedding = get_embedding(split_sentence[1], xlmr_embeddings)
    
        # First line
        n = len(sentence_list) #len(split_file)
        #assert len(sentence_list) == n, f'Not the same size: {len(sentence_list)} and {n}.'
        vec_size = embedding_size #len(np_embedding) #len(split_file[0].split(' '))
        
        with open(path, 'w', encoding = 'utf8') as out_text:
            out_text.write(f'{n} {vec_size}\n{split_sentence[0]} {" ".join(str_embedding)}\n')
            
            #{np.array2string(np_embedding, formatter={"float_kind":lambda x: "%.6f" % x})[1:-1]}')

    embedding_list = []
    #for word in sentence_list:
    for i in tqdm(range(start_i + 1, len(sentence_list))):
        sentence = sentence_list[i]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        str_embedding = get_embedding(split_sentence[1], xlmr_embeddings)
        if str_embedding: # Not None
            embedding_list.append(f'{split_sentence[0]} {" ".join(str_embedding)}')
        if i % 10000 == 0:
            with open(path, 'a', encoding = 'utf8') as out_text:
                out_text.write('\n'.join(embedding_list) + '\n')
            embedding_list = []
    # Remaining lines
    with open(path, 'a', encoding = 'utf8') as out_text:
        out_text.write('\n'.join(embedding_list) + '\n')
    #return embedding_list

def get_labse_embeddings(split_sentence, labse_model):
    '''Get the string version of the sentence embedding from the LaBSE model.'''
    labse_embedding = labse_model.encode(split_sentence)
    #print(labse_embedding, labse_embedding.shape)
    np_embedding = labse_embedding.tolist()
    #print(type(labse_embedding), len(np_embedding))
    str_embedding = [f'{embed_value:.6f}' for embed_value in labse_embedding]
    return str_embedding

def to_labse_sentence_embeddings(path, sentence_list, start_i=0):
    labse_model = SentenceTransformer('sentence-transformers/LaBSE')
    
    embedding_size = 768 #300
    
    # Initial step
    if start_i == 0:
        sentence = sentence_list[0]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        str_embedding = get_labse_embeddings(split_sentence[1], labse_model) # Convert format 
    
        # First line
        n = len(sentence_list)
        vec_size = embedding_size #len(np_embedding) #len(split_file[0].split(' '))
        
        with open(path, 'w', encoding = 'utf8') as out_text:
            out_text.write(f'{n} {vec_size}\n{split_sentence[0]} {" ".join(str_embedding)}\n')

    embedding_list = []
    #for word in sentence_list:
    for i in tqdm(range(start_i + 1, len(sentence_list))):
        sentence = sentence_list[i]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        str_embedding = str_embedding = get_labse_embeddings(split_sentence[1], labse_model) #get_embedding(split_sentence[1], xlmr_embeddings)
        if str_embedding: # Not None
            embedding_list.append(f'{split_sentence[0]} {" ".join(str_embedding)}')
        if i % 10000 == 0:
            with open(path, 'a', encoding = 'utf8') as out_text:
                out_text.write('\n'.join(embedding_list) + '\n')
            embedding_list = []
    # Remaining lines
    with open(path, 'a', encoding = 'utf8') as out_text:
        out_text.write('\n'.join(embedding_list) + '\n')

### LASER

def to_laser_sentence_embeddings(path, sentence_list, laser_model='laser2', start_i=0):
    from laser_encoders import LaserEncoderPipeline

    laser_dict = {'laser2': 'laser2', 'laser3': 'laser3-szl_Latn'} # WRONG: put szl_Latn as a specific parameter value in the Pipeline object
    encoder = LaserEncoderPipeline(laser=laser_dict[laser_model]) #'laser2')
    
    embedding_size = 1024 #768 #300
    
    # Initial step
    if start_i == 0:
        #sentence = sentence_list[0]
        #split_sentence = sentence.split('\t')
        #assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        #str_embedding = encoder.encode_sentences(split_sentence[1])
    
        # First line
        n = len(sentence_list)
        vec_size = embedding_size #len(np_embedding) #len(split_file[0].split(' '))
        
        with open(path, 'w', encoding = 'utf8') as out_text:
            out_text.write(f'{n} {vec_size}\n') #{split_sentence[0]} {" ".join(str_embedding)}\n')

    embedding_list = []
    #for word in sentence_list:
    for i in tqdm(range(start_i, len(sentence_list))): # + 1
        sentence = sentence_list[i]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        str_embedding = get_sentence_embeddings(split_sentence[1], encoder, model_name='LASER') #get_embedding(split_sentence[1], xlmr_embeddings)
        if str_embedding: # Not None
            embedding_list.append(f'{split_sentence[0]} {" ".join(str_embedding)}')
        if i % 10000 == 0:
            with open(path, 'a', encoding = 'utf8') as out_text:
                out_text.write('\n'.join(embedding_list) + '\n')
            embedding_list = []
    # Remaining lines
    with open(path, 'a', encoding = 'utf8') as out_text:
        out_text.write('\n'.join(embedding_list) + '\n')



# Extracting sentence embeddings in both languages
def main():
    args = parse_args()

    # Input file
    input_file = open(args.input_file, 'r').read()
    split_file = text_to_line(input_file)

    # In German
    #print('Extracting the German embeddings.')
    #fasttext_file_de = open('./results/embeddings/news.2011-14.de.vec', 'r').read()
    #fasttext_file_de = open('/dss/dssfs04/lwp-dss-0002/pn39je/pn39je-dss-0000/go25puh/UnsupPSE/results/embeddings/news.2011-14.de.vec', 'r').read()
    #cfw_de = collect_fasttext_words(fasttext_file_de)

    model_name = 'pretrained' #'glot500' #'xlmr'
    print(f'Model to use: {model_name}')
    if model_name in ['xlmr', 'glot500', 'pretrained']:
        to_xlmr_sentence_embeddings(args.output_file, split_file, start_i=0)
    elif model_name == 'labse':
        to_labse_sentence_embeddings(args.output_file, split_file, start_i=0)
    #'./results/new_embeddings/news.2011-14.de.vec', cfw_de, start_i=0)
    return 0

if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)

    main()