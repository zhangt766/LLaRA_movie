import pickle
from typing import Optional
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
from transformers import LlamaTokenizer, LlamaModel, AutoModelForCausalLM, LlamaForCausalLM, GenerationConfig, LlamaConfig, AutoTokenizer
# 全局加载LLaMA-2-7B模型
model_name_or_path = "/workspace/llama/models_hf/Llama-2-7b-hf"

print("Loading tokenizer...")
tokenizer: Optional[LlamaTokenizer] = AutoTokenizer.from_pretrained(model_name_or_path)
# 设置pad_token为eos_token
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[LlamaForCausalLM] = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, output_hidden_states=True)
try: model.to(device)
except : pass
print("Model loaded.")

from tqdm import tqdm
import json

def load_data(file_path):
    return pd.read_pickle(file_path)

def load_movie_dict(item_file):
    item_df = pd.read_csv(item_file, sep='::', header=None, encoding='latin-1', engine='python', usecols=[0, 1])
    item_df.columns = ['movie_id', 'movie_title']
    movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))
    return movie_dict

def map_movie_names_only(seq, movie_dict):
    return [movie_dict[id] if id in movie_dict else id for id in seq]

def extract_sequences(df, movie_dict):
    df['movie_names_only'] = df['seq'].apply(lambda x: map_movie_names_only(x, movie_dict))
    return df

def get_movie_embeddings(movie_list):
    embeddings = []
    max_length = 512  # 设定一个合理的最大长度
    for movies in tqdm(movie_list):
        movie_string = " ".join(str(movie) for movie in movies)
        inputs = tokenizer(movie_string, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            movie_embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()
        embeddings.append(movie_embedding)
    return torch.stack(embeddings)
def get_topk_similar_indices(similarity_scores, topK):
    print(similarity_scores.shape)
    indices = np.argsort(-np.array(similarity_scores.to(torch.float32)))
    print(indices.shape)
    print(indices[-5:])
    topk_indices = np.ones((indices.shape[0], topK))
    for i,indice in enumerate(indices):
        tmp = indice[indice!=i]
        topk_indices[i] = tmp[:topK] # 获取每个向量最相似的topK个索引, 不包含他自己
    # topk_indices = topk_indices.to(torch.int)
    print(topk_indices.shape)
    return topk_indices

def get_topK_candidate(df, topK=10):
    embeddings = get_movie_embeddings(df['movie_names_only'].tolist())
    # df['movie_embeddings'] = list(movie_embeddings)
    # embeddings = np.stack(df['movie_embeddings'].values)
    similarity_scores = embeddings @ embeddings.T
    # 对于每个嵌入向量，找到最相似的topK个嵌入向量的索引
    most_similar_indices = np.array(get_topk_similar_indices(similarity_scores, topK)).tolist()
    print(type(most_similar_indices))
    # 将索引信息添加到DataFrame中
    df['most_similar_seq_index'] = [json.dumps(most_similar_idxs) for most_similar_idxs in most_similar_indices]
    # 根据索引获取最相似的序列
    df['most_similar_seq'] = df['most_similar_seq_indexs'].apply(lambda idxs: [df.at[idx, 'seq'] for idx in json.loads(idxs)])
    return df

def add_most_similar_seq_next(df, movie_dict):
    df['most_similar_seq_next'] = df['most_similar_seq_index'].apply(lambda idxs: [df.at[int(idx), 'next'] for idx in json.loads(idxs)])
    df['most_similar_seq_name'] = df['most_similar_seq'].apply(lambda seqs: [[movie_dict.get(item, "Unknown") for item in items] for items in seqs])
    df['most_similar_seq_next_name'] = df['most_similar_seq_next'].apply(lambda nexts: [movie_dict.get(item, "Unknown") for item in nexts])
    return df


def save_data(df, output_file_path):
    df.to_pickle(output_file_path)

def process_data(file_path, item_file, output_file_path):
    data = load_data(file_path)
    movie_dict = load_movie_dict(item_file)
    df = extract_sequences(data, movie_dict)
    df = get_topK_candidate(df)
    df = add_most_similar_seq_next(df, movie_dict)
    save_data(df, output_file_path)
    return df

file_path = '/workspace/LLaRA/data/ref/lastfm/train_data.df'
item_file = '/workspace/LLaRA/data/ref/lastfm/id2name.txt'
output_file_path = '/workspace/LLaRA/data/ref/lastfm/similar_train_data.df'

