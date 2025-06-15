import fasttext, numpy as np, os
from math import ceil
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir="./models")
model = fasttext.load_model(model_path)
import torch
device = ""
if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
labse_encoder = SentenceTransformer('sentence-transformers/LaBSE', device=device)
lang_codes = {"english": "__label__eng_Latn", "sinhala": "__label__sin_Sinh"}
lang_laser_encoders = {}
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])
from laser_encoders import LaserEncoderPipeline
for language in lang_codes.keys():
    lang_laser_encoders[language] = LaserEncoderPipeline(lang=language)
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
furina_path = "yihongLiu/furina" 
furina_indic_path = "yihongLiu/furina-indic"       
furina = EmbeddingLoader(furina_path, torch.device(device=device), layer=8)
furina_indic = EmbeddingLoader(furina_indic_path, torch.device(device=device), layer=8)

LENGTH_RATIO_MEAN = 0.968442560084747
LENGTH_RATIO_STD = 0.2514354396303809
#Define length ratio parameters based on NLLB
with open("data/en-si/NLLB.en-si.en") as f1, open("data/en-si/NLLB.en-si.si") as f2: 
    ratios = []
    for line1, line2 in zip(f1, f2):
        line1 = line1.removesuffix("\n")
        line2= line2.removesuffix("\n")
        ratios.append(float(len(line1)/len(line2)))
    import statistics 
    LENGTH_RATIO_MEAN = statistics.fmean(ratios)
    LENGTH_RATIO_STD = statistics.stdev(ratios)
#     print(str(LENGTH_RATIO_MEAN) + " " + str(LENGTH_RATIO_STD))

def preprocess_line(line):
    line = line.removesuffix("\n")
    line = line.replace("\u200c", "")
    line = line.replace("\u200d", "")
    return line

#Check if the lengths of the sentence pairs match:
def check_lengths(df, lang1, lang2, z_thresh=2.326):
    #Current default z_thresh is for a 98% confidence interval
    import numpy as np
    ratios = df[f"{lang1}"].apply(lambda x: len(x.split())) / df[f"{lang2}"].apply(lambda x: len(x.split()) if len(x.split()) > 0 else 1)
    z_scores = list(map(lambda ratio : ((ratio - LENGTH_RATIO_MEAN) / LENGTH_RATIO_STD), ratios))
    return df[np.abs(z_scores) <= z_thresh].reset_index(drop=True)

def check_script(sentence, lang):
    from GlotScript import sp, sc 
    GlotScript_codes = {"english": "Latn", "sinhala": "Sinh"}
    result = sc(sentence)
    if GlotScript_codes[lang] in result.keys() and len(result[GlotScript_codes[lang]])/len(sentence) >= 0.2:
        return True
    else:
        return False 

def check_scripts(df, langs):
    lang1_mask = df[f"{langs[0]}"].apply(check_script, args=(langs[0],)) 
    lang2_mask = df[f"{langs[1]}"].apply(check_script, args=(langs[1],))
    return df[lang1_mask & lang2_mask].reset_index(drop=True)

#Return true if the sentence belongs to the specified language
# def check_language(sentence, language_code, threshold):
#     x = model.predict(sentence)
#     if str(x[0][0]) == language_code and float(x[1][0]) >= threshold:
#         return True 
#     else:
#         return False

# def check_languages(df, langs):
#     lang1_mask = df[f"{langs[0]}"].apply(check_language, args=(lang_codes[f"{langs[0]}"], 0.5))
#     lang2_mask = df[f"{langs[1]}"].apply(check_language, args=(lang_codes[f"{langs[1]}"], 0.5))
#     return df[lang1_mask & lang2_mask].reset_index(drop=True)

def check_languages(df, langs):
    lang1_scores = []
    lang2_scores = []
    for sentence in list(df[langs[0]]):
        x = model.predict(sentence, 10)
        if lang_codes[langs[0]] in list(map(lambda e: str(e), list(x[0]))):
            lang1_scores.append(float(x[1][list(x[0]).index(lang_codes[langs[0]])]))
        else:
            lang1_scores.append(0)
    for sentence in list(df[langs[1]]):
        x = model.predict(sentence, 10)
        if lang_codes[langs[1]] in list(map(lambda e: str(e), list(x[0]))):
            lang2_scores.append(float(x[1][list(x[0]).index(lang_codes[langs[1]])]))
        else:
            lang2_scores.append(0) 
    df[f"{langs[0]} score"] = lang1_scores
    df[f"{langs[1]} score"] = lang2_scores
    df = static_filter(df, f"{langs[0]} score", 0.5)
    df = static_filter(df, f"{langs[1]} score", 0.5)
    return df 

#Convert Moses format files into a pandas DataFrame
def moses_to_df(file1, file2, lang1, lang2):
    lang1_lines = []
    lang2_lines = []
    #Read Moses files into a Pandas DataFrame
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        # i = 0
        for line1, line2 in list(zip(f1, f2)): 
            # if i==512:
            #     break 
            line1 = preprocess_line(line1)
            line2 = preprocess_line(line2)
            if line1 == "" or line2 == "":
                continue 
            lang1_lines.append(line1)
            lang2_lines.append(line2)
            # i = i +1 
    import pandas as pd
    df = pd.DataFrame({lang1: lang1_lines, lang2: lang2_lines})
    return df

def tmx_to_df(file, lang1, lang2):
    import xml.etree.ElementTree as ET
    # tmx_codes = {"english": "en", "sinhala": "si"}
    lang1_lines = []
    lang2_lines = []
    tree = ET.parse(file)
    root = tree.getroot()
    body = root.find('body')
    # i = 0
    for tu in body.findall('tu'): #[115651:]:
        # if i==512:
        #     break
        src_seg = tu[0][0]
        tgt_seg = tu[1][0] 
        if src_seg is None or tgt_seg is None: 
            continue
        if src_seg.text.strip() == "" or tgt_seg.text.strip() == "":
            continue 
        source_text = src_seg.text
        target_text = tgt_seg.text 
        source_text = preprocess_line(source_text)
        target_text = preprocess_line(target_text)
        lang1_lines.append(source_text)
        lang2_lines.append(target_text)
        # i = i +1 
    import pandas as pd
    df = pd.DataFrame({lang1: lang1_lines, lang2: lang2_lines})
    return df

def xlmr_encode(sentences, model):
    '''Getting an embedding from XLMR.'''
    embeddings = []
    for sentence in sentences:
        s_embedding = model.get_embed_list([sentence])
        if s_embedding.size(dim=1) > 1: # More than 1 dimension of the second dimension
            np_embedding = s_embedding.cpu().detach().numpy()[0].mean(axis=0)
            #print(np_embedding.shape)
        elif s_embedding.size(dim=1) == 1: # Only one dimension
            np_embedding = s_embedding.cpu().detach().numpy()[0][0]
        else: # Continue
            continue
        embeddings.append(np_embedding)
    embeddings = np.asarray(embeddings)
    return embeddings

#Convert a list of sentences into their multilingual embeddings according to the given model
def to_multilingual_embedding(language, sentences, model):
    import torch
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if model.lower() == "labse":
        embedding = labse_encoder.encode(sentences, device=device)
    if model.lower() == "laser":
        embedding = lang_laser_encoders[language].encode_sentences(sentences)
    if model.lower() == "furina":
        embedding = xlmr_encode(sentences, furina)
    if model.lower() == "furina-indic":
        embedding = xlmr_encode(sentences, furina_indic)
    return embedding

#Find similarity scores for sentence pairs using cosine similarity
def find_similarity_score(embeddings1, embeddings2): 
    import statistics
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings1, embeddings2)
    similarity_scores = [statistics.fmean(vector) for vector in similarities]
    # similarity_scores = [float(vector.sum()) for vector in similarities]
    return similarity_scores

#Given a pandas DataFrame, filter best x percent of sentence pairs and store the results in a .tsv file
def distribution_filter(df, column): 
    import numpy as np
    hist, bin_edges = np.histogram(list(df[column]), bins=100)
    peak_bin_index = np.argmax(hist)
    mode = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
    df.sort_values(column, ascending=False, inplace=True)
    df = df[df[column] >= mode]
    return df

def percentile_filter(df, column, percentile=75):
    import numpy as np
    data = df[column]
    threshold = np.percentile(data, percentile)
    df = df[df[column] >= threshold]
    return df

def static_filter(df, column, threshold):
     df = df[df[column] >= threshold]
     return df 

def word_alignment_filter(df, langs):
    import scripts.align_source_target as ast
    source_lines = df[langs[0]]
    target_lines = df[langs[1]]
    margin_lines = list(df.index)
    src_file = ""
    trg_file=""
    mrg_file = ""
    for id, sentence in zip(margin_lines, source_lines):
        src_file = src_file + f"{id}" + "\t" + sentence + "\n"
    for id, sentence in zip(margin_lines, target_lines):
        trg_file = trg_file + f"{id}" + "\t" + sentence + "\n"
    for id, sentence in zip(margin_lines, source_lines):
        mrg_file = mrg_file + f"{id}" + "\t" + f"{id}" + "\n"
    src_file_dict = ast.text_to_dict(src_file)
    trg_file_dict = ast.text_to_dict(trg_file)
    margin_train_file = mrg_file
    split_margin_train = margin_train_file.split('\n')[:-1]
    align_list_train = ast.align_source_target(split_margin_train, src_file_dict, trg_file_dict) 
    align_rate_train_file = ast.align_rate_file(split_margin_train, align_list_train, src_file_dict, trg_file_dict) 
    alignment_score = []
    for line in align_rate_train_file:
        split_align = line.split("\t")
        alignment_score.append(float(split_align[2]))
    df["alignment score"] = alignment_score
    df = static_filter(df, "alignment score", 0.3)
    return df

def tsv_to_moses_files(file):
    prefix = file.removesuffix(".tsv").removeprefix("outputs/")
    with open(file, "r", encoding="utf-8") as f:
        en_lines = []
        si_lines = []    
        scores = []
        for line in f:
            split = line.split("\t")
            en_lines.append(preprocess_line(split[1]))
            si_lines.append(preprocess_line(split[2]))
            scores.append(preprocess_line(split[3]))
    with open(f"training_nmt/{prefix}.en", "w", encoding="utf-8") as f:
        for line in en_lines[1:]:
            f.write(f"{line}\n")
    with open(f"training_nmt/{prefix}.si", "w", encoding="utf-8") as f:
        for line in si_lines[1:]:
            f.write(f"{line}\n")
    with open(f"training_nmt/{prefix}.scores.txt", "w", encoding="utf-8") as f:
        for line in scores[1:]:
            f.write(f"{line}\n")

def main(files, langs, output, model, type="moses"):
    import pandas as pd
    df = None
    filtering_stats = {}
    if type=="moses":
        df = moses_to_df(files[0], files[1], langs[0], langs[1])
    if type=="tmx":
        df = tmx_to_df(files[0], langs[0], langs[1]) 
    filtering_stats["Raw corpus"] = df.shape[0]
    #Remove duplicated sentence pairs 
    df.drop_duplicates(inplace=True, ignore_index=True)
    filtering_stats["After dropping duplicates"] = df.shape[0]
    #Remove pairs where the ratios of words per sentence is too unlikely
    df = check_lengths(df, langs[0], langs[1])
    filtering_stats["After removing length based outliers"] = df.shape[0]
    #Remove pairs where one of the sentences is in the wrong script
    df = check_scripts(df, langs)
    filtering_stats["After performing script identification"] = df.shape[0]
    #Remove pairs where one of the sentences is in the wrong language
    df = check_languages(df, langs)
    filtering_stats["After performing language identification"] = df.shape[0]
    #Calculate and filter according to the similarity scores for the sentence pairs
   
    BATCH_SIZE = 512  

    all_scores = []
    total = df.shape[0]
    num_batches = ceil(total / BATCH_SIZE)

    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total)
        
        batch_df = df.iloc[start:end]
        
        lang1_embedding = to_multilingual_embedding(langs[0], batch_df[langs[0]].tolist(), model)
        lang2_embedding = to_multilingual_embedding(langs[1], batch_df[langs[1]].tolist(), model)
        
        scores = find_similarity_score(lang1_embedding, lang2_embedding)
        all_scores.extend(scores)

    df["similarity score"] = all_scores
    df = distribution_filter(df, "similarity score")
    filtering_stats["After filtering based on similarity scores"] = df.shape[0]
    #Filter according to word alignment scores for the sentence pairs
    # df = df[df[langs[0]].str.strip() != ""]
    # df = df[df[langs[1]].str.strip() != ""]
    df = df.dropna(subset=[langs[0], langs[1]])
    df = df.reset_index(drop=True)
    df= word_alignment_filter(df, langs)
    filtering_stats["After filtering based on word alignment"] = df.shape[0]
    #Create .tsv file with filtered output
    df.to_csv(output, sep="\t")
    #Print filtering stats
    for item in filtering_stats.keys():
        print(item + f": {filtering_stats[item]}\n")

def main_cli(): 
    parser = argparse.ArgumentParser("filtering.py")
    parser.add_argument("--files", "-f", type=str, nargs="+")
    parser.add_argument("--type", "-t", type=str, required=False)
    parser.add_argument("--langs", "-l", type=str, nargs="+")
    parser.add_argument("--output", "-o", type=str)
    # parser.add_argument("--percentile", "-p", type=float)
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    if args.type=="tmx":
        main(args.files, args.langs, args.output, args.model, args.type)
    else: 
        main(args.files, args.langs, args.output, args.model, args.type)
if __name__ == "__main__":
    main_cli()