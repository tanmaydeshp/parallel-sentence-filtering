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
lang_codes = {"english": "__label__eng_Latn", "sinhala": "__label__sin_Sinh", "nepali": "__label__npi_Deva"}
lang_suffixes = {"en": "english", "si": "sinhala", "ne": "nepali"}
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
def set_NLLB_ratios(lang1, lang2):
    colab_ratios = {"en, ne": (0.9693840181280292, 0.2700320861142401), 
                     "ne, en": (1.1109355531682032, 0.31124711658334797),
                     "en, si": (0.968442560084747, 0.2514354396303809), 
                     "si, en": (1.1030591942688206, 0.2932328170981356)}
    global LENGTH_RATIO_MEAN, LENGTH_RATIO_STD
    other = ""
    if lang1 != "en":
         other = lang1 
    else: 
         other = lang2 
    if not(os.path.exists(f"data/en-{other}/NLLB.en-{other}.{lang1}")) or not(os.path.exists(f"data/en-{other}/NLLB.en-{other}.{lang2}")):       
         LENGTH_RATIO_MEAN = colab_ratios[f"{lang1}, {lang2}"][0]
         LENGTH_RATIO_STD = colab_ratios[f"{lang1}, {lang2}"][1]
         return LENGTH_RATIO_MEAN, LENGTH_RATIO_STD
    else: 
        with open(f"data/en-{other}/NLLB.en-{other}.{lang1}") as f1, open(f"data/en-{other}/NLLB.en-{other}.{lang2}") as f2: 
            ratios = []
            for line1, line2 in zip(f1, f2):
                line1 = line1.removesuffix("\n")
                line2= line2.removesuffix("\n")
                ratios.append(float(len(line1)/len(line2)))
        import statistics 
        LENGTH_RATIO_MEAN = statistics.fmean(ratios)
        LENGTH_RATIO_STD = statistics.stdev(ratios)
        return LENGTH_RATIO_MEAN, LENGTH_RATIO_STD

def preprocess_line(line):
    line = line.removesuffix("\n")
    line = line.replace("\u200c", "")
    line = line.replace("\u200d", "")
    return line

def dedup(df, langs):
    df.sort_values("similarity score", ascending=False, inplace=True)
    df.drop_duplicates(subset=[f"{langs[0]}"], inplace=True)
    return df

#Check if the lengths of the sentence pairs match:
def check_lengths(df, lang1, lang2, z_thresh=2.326):
    #Current default z_thresh is for a 98% confidence interval
    import numpy as np
    ratios = df[f"{lang1}"].apply(lambda x: len(x.split())) / df[f"{lang2}"].apply(lambda x: len(x.split()) if len(x.split()) > 0 else 1)
    z_scores = list(map(lambda ratio : ((ratio - LENGTH_RATIO_MEAN) / LENGTH_RATIO_STD), ratios))
    return df[np.abs(z_scores) <= z_thresh].reset_index(drop=True)

# def check_script(sentence, lang):
#     from GlotScript import sp, sc 
#     GlotScript_codes = {"english": "Latn", "sinhala": "Sinh", "nepali": "Deva"}
#     result = sc(sentence)
#     # script_score = sp(sentence)[1]
#     if GlotScript_codes[lang] in result.keys() and len(result[GlotScript_codes[lang]])/len(sentence) >= 0.2:
#         print(len(result[GlotScript_codes[lang]])/len(sentence))
#         return True
#     else:
#         return False 

def check_scripts(df, langs):
    import pandas as pd 
    lang1_sentences = df[langs[0]]
    lang2_sentences = df[langs[1]]
    lang1_mask = []
    lang2_mask = []
    lang1_scores = []
    lang2_scores = []
    from GlotScript import sp, sc 
    GlotScript_codes = {"english": "Latn", "sinhala": "Sinh", "nepali": "Deva"}
    for s1, s2 in zip(lang1_sentences, lang2_sentences):
         result1 = sc(s1)
         result2 = sc(s2)
         if GlotScript_codes[langs[0]] in result1.keys():
            score1 = sp(s1)[2]["details"][GlotScript_codes[langs[0]]]
            lang1_scores.append(score1)
         else:
            lang1_scores.append(0)
         if GlotScript_codes[langs[1]] in result2.keys():
            score2 = sp(s2)[2]["details"][GlotScript_codes[langs[1]]]
            lang2_scores.append(score2)
         else:
            lang2_scores.append(0)
    df[f"{langs[0]} script score"] = lang1_scores 
    df[f"{langs[1]} script score"] = lang2_scores
    # for score in lang1_scores:
    #     if score >= 0.7:
    #         lang1_mask.append(True)
    #     else:
    #         lang1_mask.append(False)
    # for score in lang2_scores:
    #     if score >= 0.7:
    #         lang2_mask.append(True)
    #     else:
    #         lang2_mask.append(False)
    # return df[pd.Series(lang1_mask) & pd.Series(lang2_mask)]
    return df

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
    df[f"{langs[0]} language score"] = lang1_scores
    df[f"{langs[1]} language score"] = lang2_scores
    # df = static_filter(df, f"{langs[0]} language score", 0.5)
    # df = static_filter(df, f"{langs[1]} language score", 0.5)
    #df = distribution_filter(df, f"{langs[0]} score")
    #df = distribution_filter(df, f"{langs[1]} score")
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
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings1, embeddings2)
    similarity_scores = np.diag(similarities).tolist()
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

def statistical_filter(df, column):
    scores = list(df[column])
    import statistics
    mean = statistics.fmean(scores)
    std = statistics.stdev(scores)
    threshold = mean - std
    df.sort_values(column, ascending=False, inplace=True)
    df = df[df[column] >= threshold]
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

def word_alignment_filter(df, langs, outdir):
    import align_source_target as ast
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
    # df = static_filter(df, "alignment score", 0.3)
    #df = distribution_filter(df, "alignment score")
    df.to_csv(f"{outdir}/aligned.tsv", sep="\t", index=None)
    return df

def batch_aligns(start, end, langs, outdir):
    import pandas as pd 
    df = pd.read_csv(f"{outdir}/prealigned.tsv", sep="\t")
    df = df[start:end]
    import align_source_target as ast
    source_lines = df[langs[0]]
    target_lines = df[langs[1]]
    margin_lines = list(range(len(df)))
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
    with open(f"{outdir}/alignments.txt", "a+", encoding="utf-8") as file:
        for alignment in alignment_score:
            file.write(f"{alignment}\n")
    return alignment_score

def tsv_to_moses_files(file, suffix1, suffix2):
    prefix = file.removesuffix(".tsv").removeprefix("outputs/")
    with open(file, "r", encoding="utf-8") as f:
        lang1_lines = []
        lang2_lines = []    
        scores = []
        for line in f:
            split = line.split("\t")
            lang1_lines.append(preprocess_line(split[1]))
            lang2_lines.append(preprocess_line(split[2]))
            scores.append(preprocess_line(split[3]))
    with open(f"{prefix}.{suffix1}", "w", encoding="utf-8") as f:
        for line in lang1_lines[1:]:
            f.write(f"{line}\n")
    with open(f"{prefix}.{suffix2}", "w", encoding="utf-8") as f:
        for line in lang2_lines[1:]:
            f.write(f"{line}\n")
    with open(f"{prefix}.scores.txt", "w", encoding="utf-8") as f:
        for line in scores[1:]:
            f.write(f"{line}\n")

def tsv_to_prefiltering_scores(tsv, langs, output):
    langs_suffixes = {"english":"en", "sinhala": "si", "nepali":"ne"}
    import pandas as pd 
    df = pd.read_csv(tsv, sep="\t")
    lang1_script_scores = df[f"{langs[0]} script score"]
    lang2_script_scores = df[f"{langs[1]} script score"]
    lang1_language_scores = df[f"{langs[0]} language score"]
    lang2_language_scores = df[f"{langs[1]} language score"]
    scores = (lang1_script_scores + lang2_script_scores + lang1_language_scores + lang2_language_scores)/4
    df["prefiltered score"] = scores 
    df.sort_values(by="prefiltered score", inplace=True, ascending=False)
    lang1_sentences = df[f"{langs[0]}"]
    lang2_sentences = df[f"{langs[1]}"]
    with open(f"{output}/prefiltered_lang1.{langs_suffixes[langs[0]]}", "w", encoding="utf-8") as f1:
        for line in lang1_sentences:
            f1.write(f"{line}\n")
    with open(f"{output}/prefiltered_lang2.{langs_suffixes[langs[1]]}", "w", encoding="utf-8") as f2:
        for line in lang2_sentences:
            f2.write(f"{line}\n")
    with open(f"{output}/prefiltered_scores.txt", "w", encoding="utf-8") as file:
        for score in df["prefiltered score"]: 
            file.write(f"{score}\n")

def tsv_to_final_scores(tsv, langs, output):
    langs_suffixes = {"english":"en", "sinhala": "si", "nepali":"ne"}
    import pandas as pd 
    df = pd.read_csv(tsv, sep="\t")
    similarity_scores = df["similarity score"]
    alignment_scores = df["alignment score"]
    lang1_script_scores = df[f"{langs[0]} script score"]
    lang2_script_scores = df[f"{langs[1]} script score"]
    lang1_lang_scores = df[f"{langs[0]} language score"]
    lang2_lang_scores = df[f"{langs[1]} language score"]
    scores = (0.7 * similarity_scores + 0.3 * alignment_scores)
    df["final score"] = scores 
    df.sort_values(by="final score", inplace=True, ascending=False)
    lang1_sentences = df[f"{langs[0]}"]
    lang2_sentences = df[f"{langs[1]}"]
    with open(f"{output}/final_lang1.{langs_suffixes[langs[0]]}", "w", encoding="utf-8") as f1:
        for line in lang1_sentences:
            f1.write(f"{line}\n")
    with open(f"{output}/final_lang2.{langs_suffixes[langs[1]]}", "w", encoding="utf-8") as f2:
        for line in lang2_sentences:
            f2.write(f"{line}\n")
    with open(f"{output}/final_scores.txt", "w", encoding="utf-8") as file:
        for score in df["final score"]: 
            file.write(f"{score}\n")


def return_prefiltered(files, outdir):
    type = ""
    df = None
    filtering_stats = {}
    if len(files) == 1:
         type = "tmx"
    else: 
         type = "moses"
    langs = []
    if type=="tmx": 
        import xml.etree.ElementTree as ET
        tree = ET.parse(files[0])
        root = tree.getroot()
        body = root.find('body')
        tu = body.find('tu') 
        lang1 = tu[0].get("{http://www.w3.org/XML/1998/namespace}lang")
        lang2 = tu[1].get("{http://www.w3.org/XML/1998/namespace}lang")
        langs.append(lang_suffixes[lang1])
        langs.append(lang_suffixes[lang2])
        df = tmx_to_df(files[0], langs[0], langs[1]) 
    if type == "moses":
         lang1 = files[0].split(".")[-1]
         lang2 = files[1].split(".")[-1]
         langs.append(lang_suffixes[lang1])
         langs.append(lang_suffixes[lang2])
         df = moses_to_df(files[0], files[1], langs[0], langs[1])
    set_NLLB_ratios(lang1, lang2)
    #Raw corpus size 
    filtering_stats["Raw corpus size"] = df.shape[0]
    #Remove duplicated sentence pairs 
    # df.drop_duplicates(inplace=True, ignore_index=True)
    # filtering_stats["After dropping duplicates"] = df.shape[0]
    #Remove pairs where the ratios of words per sentence is too unlikely
    # df = check_lengths(df, langs[0], langs[1])
    # filtering_stats["After removing length based outliers"] = df.shape[0]
    #Remove boilerplate
    # df = remove_boilerplate(df, langs)
    #Remove pairs where one of the sentences is in the wrong script
    df = check_scripts(df, langs)
    filtering_stats["After performing script identification"] = df.shape[0]
    #Remove pairs where one of the sentences is in the wrong language
    df = check_languages(df, langs)
    filtering_stats["After performing language identification"] = df.shape[0]
    df.to_csv(f"{outdir}/prefiltered.tsv", sep="\t", index=None)
    return df, langs, filtering_stats 

def batch_embeds(start, end, langs, model, outdir):
    import pandas as pd 
    df = pd.read_csv(f"{outdir}/prefiltered.tsv", sep="\t")
    total = df.shape[0]
    df = df[start:end]
    lang1_lines = list(df[f"{langs[0]}"])
    lang2_lines = list(df[f"{langs[1]}"])
    ids = list(df.index) 
    with open(f"{outdir}/embeddings.tsv", "a+", encoding="utf-8") as file:
        e1 = to_multilingual_embedding(langs[0], lang1_lines, model)
        e2 = to_multilingual_embedding(langs[1], lang2_lines, model)
        similarity_scores = find_similarity_score(e1, e2)
        for score in similarity_scores:
            file.write(f"{score}\n")
    with open(f"{outdir}/{langs[0]}_embeds.vec", "a+", encoding="utf-8") as f:
        if start == 0:
            f.write(f"{total} {e1.shape[1]}\n")
        for id, encoding in zip(ids, e1):
            encoding_str = " ".join([str(x) for x in encoding])
            f.write(f"src-{id} {encoding_str}\n")
    with open(f"{outdir}/{langs[1]}_embeds.vec", "a+", encoding="utf-8") as f:
        if start == 0:
            f.write(f"{total} {e2.shape[1]}\n")
        for id, encoding in zip(ids,e2):
            encoding_str = " ".join([str(x) for x in encoding])
            f.write(f"tgt-{id} {encoding_str}\n")
    return total, e1.shape[1]

def csls_adjustment(outdir, langs):
    import pandas as pd
    import bilingual_nearest_neighbor as bnn 
    import torch 
    bnn.main(source_embeddings=f"{outdir}/{langs[0]}_embeds.vec", 
            target_embeddings=f"{outdir}/{langs[1]}_embeds.vec", 
            output=f"{outdir}/st_embeds.txt", 
            binary=0, method="csls", knn=10, cslsknn=10, gpu=torch.cuda.is_available)
    df = pd.read_csv(f"{outdir}/prefiltered.tsv", sep="\t")
    csls_scores = []
    with open(f"{outdir}/st_embeds.txt", "r", encoding="utf-8") as f:
        for line in f: 
            split = line.split("\t")
            src_id = int(preprocess_line(split[0].split("-")[-1]))
            split = split[1:]
            tgt_ids = []
            tgt_scores = []
            idx = -1 
            for i in range(0, len(split) -1, 2):
                tgt_ids.append(int(preprocess_line(split[i].split("-")[-1])))
                tgt_scores.append(float(preprocess_line(split[i+1])))
            for j in range(0, len(tgt_ids)):
                if tgt_ids[j] == src_id:
                    score = tgt_scores[j]
                    idx = j
                    csls_scores.append(score)
                    break 
            if idx==-1:
                import statistics
                mean = statistics.fmean(tgt_scores)
                csls_scores.append(mean)       
    df["similarity score"] = csls_scores
    df.to_csv(f"{outdir}/prealigned.tsv", sep="\t", index=None)
    alignments = []
    with open(f"{outdir}/alignments.txt", "r", encoding="utf-8") as f: 
        for line in f: 
            alignments.append(float(preprocess_line(line)))
    df["alignment score"] = alignments
    df.to_csv(f"{outdir}/aligned.tsv", sep="\t", index=None)

def merge_phrases(common_phrases):
    from GlotScript import sp    
    merged = set()
    phrases = []
    for phrase in common_phrases:
        if phrase[1] >= 500 and sp(phrase[0])[0] != "Deva":
            phrases.append(phrase[0])
    for i in range(len(phrases)):
        included = False 
        for j in range(len(phrases)):
            if phrases[i] in phrases[j] and i!=j:
                included = True
        if not included:
            merged.add(phrases[i])
    return merged 

    
def find_common_phrases(sentences, top_k):
    from collections import Counter 
    phrases = []
    for sentence in sentences: 
        tokens = sentence.split()
        num_words = len(tokens)
        if num_words >= 4:
            phrases.append(" ".join(tokens[:4]))
            phrases.append(" ".join(tokens[-4:]))
        if num_words >=3:
            phrases.append(" ".join(tokens[:3]))
            phrases.append(" ".join(tokens[-3:]))
        if num_words >= 2:
            phrases.append(" ".join(tokens[:2]))
            phrases.append(" ".join(tokens[-2:]))
    return Counter(phrases).most_common(top_k)

def remove_affix(sentence, phrase):
    if sentence.startswith(phrase):
        sentence = sentence.removeprefix(phrase)
    if sentence.endswith(phrase):
        sentence = sentence.removesuffix(phrase)
    return sentence

def boilerplate_filter(df, column, top_k=50): 
    import numpy as np    
    df = df.dropna()
    i = 0
    sentences = list(df[column])
    new_sentences = []
    for sentence in sentences: 
        new_tokens = []
        tokens = sentence.split()
        for token in tokens: 
            if token.isdigit():
                new_tokens.append("<num>")
            else:
                new_tokens.append(token)
        new_sentence = " ".join(new_tokens)
        new_sentences.append(new_sentence)
    common_phrases = find_common_phrases(new_sentences, top_k)
    merged_phrases = merge_phrases(common_phrases)
    print(merged_phrases)
    filtered = []
    for sentence in new_sentences:
        for phrase in merged_phrases:
                sentence = sentence.strip().removeprefix(phrase)
                sentence = sentence.strip().removesuffix(phrase)
        if sentence == "":
            filtered.append(np.nan)
        else:
            filtered.append(sentence)
    df[column] = filtered 
    df = df.dropna()
    return df 
                
def remove_boilerplate(df, langs): 
    df = boilerplate_filter(df, langs[0])
    df = boilerplate_filter(df, langs[1])
    return df 

def main(files, output, model, csls):
    import json 
    import os
    import shutil
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    df, langs, filtering_stats = return_prefiltered(files, output)
    tsv_to_prefiltering_scores(f"{output}/prefiltered.tsv", langs, output)
    #Calculate and filter according to the similarity scores for the sentence pairs
    BATCH_SIZE = 1024 
    for i in range(0, df.shape[0], BATCH_SIZE):
        batch_embeds(i, i + BATCH_SIZE, langs, model, output)
    similarity_scores = []
    if(csls):
        import bilingual_nearest_neighbor as bnn 
        bnn.main(source_embeddings=f"{output}/{langs[0]}_embeds.vec", 
                target_embeddings=f"{output}/{langs[1]}_embeds.vec", 
                output=f"{output}/st_embeds.txt", 
                binary=0, method="csls", knn=50, cslsknn=50, gpu=torch.cuda.is_available)
        csls_scores = []
        with open(f"{output}/st_embeds.txt", "r", encoding="utf-8") as f:
            for line in f: 
                split = line.split("\t")
                src_id = float(split[0].split("-")[-1])
                split = split[1:]
                tgt_ids = []
                tgt_scores = []
                idx = -1 
                for i in range(0, len(split) -1, 2):
                    tgt_ids.append(float(preprocess_line(split[i].split("-")[-1])))
                    tgt_scores.append(float(preprocess_line(split[i+1])))
                for j in range(0, len(tgt_ids)):
                    if tgt_ids[j] == src_id:
                        score = tgt_scores[j]
                        idx = j
                        csls_scores.append(score)
                        break 
                if idx==-1:
                    csls_scores.append(0.0)
                
        similarity_scores = csls_scores
    else:
        cosine_scores = []
        with open(f"{output}/embeddings.tsv", "r", encoding="utf-8") as file: 
            for score in file:
                cosine_scores.append(float(preprocess_line(score)))
        similarity_scores = cosine_scores
    df["similarity score"] = similarity_scores
    df.sort_values("similarity score", ascending=False, inplace=True)
    df.to_csv(f"{output}/prealigned.tsv",sep="\t",index=None)
    filtering_stats["After filtering based on similarity scores"] = df.shape[0]
    df = df.dropna(subset=[langs[0], langs[1]])
    df = df.reset_index(drop=True)
    # alignments = []
    # for i in range(0, df.shape[0], BATCH_SIZE): 
    #     align_scores = batch_aligns(df[i:i+BATCH_SIZE], langs, output)
    #     alignments.extend(align_scores)
    # df["alignment score"] = alignments
    alignments = []
    for i in range(0, df.shape[0], BATCH_SIZE): 
        align_scores = batch_aligns(i, i + BATCH_SIZE, langs, output)
        alignments.extend(align_scores)
    df["alignment score"] = alignments
    # df = static_filter(df, "alignment score", 0.3)
    #df = distribution_filter(df, "alignment score")
    df.to_csv(f"{output}/aligned.tsv", sep="\t", index=None)
    filtering_stats["After filtering based on word alignment"] = df.shape[0]
    df.sort_values("similarity score", ascending=False, inplace=True)
    #Remove multiple translations of the same sentence 
    df = dedup(df, langs)
    filtering_stats["After removing multiple translations"] = df.shape[0]
    df.to_csv(f"{output}/final.tsv", sep="\t", index=None)
    tsv_to_final_scores(f"{output}/final.tsv", langs, output)
    #Print filtering stats
    for item in filtering_stats.keys():
        print(item + f": {filtering_stats[item]}\n")
    json_name = output.split("/")[-1].removesuffix(".tsv")
    with open(f"stats/{json_name}.json", "w", encoding="utf-8") as f:
         f.write(json.dumps(filtering_stats))

def main_cli(): 
    parser = argparse.ArgumentParser("filtering.py")
    parser.add_argument("--files", "-f", type=str, nargs="+")
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--csls", "-c", type=bool, required=False, default=False)
    args = parser.parse_args()
    main(args.files, args.output, args.model, args.csls)
if __name__ == "__main__":
    main_cli()