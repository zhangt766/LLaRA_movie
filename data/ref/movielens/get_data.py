# # import pickle
# # import pandas as pd
# # file_path = '/data/projects/wsx/LLaRA/data/ref/movielens/Test_data.df'
# # df = pd.read_pickle(file_path)


# # # 读取u.item文件
# # item_file = '/data/projects/wsx/LLaRA/data/ref/movielens/u.item'
# # item_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', usecols=[0, 1])
# # item_df.columns = ['movie_id', 'movie_title']


# # # 创建字典来映射电影 ID 到电影名称
# # movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))


# # # 添加电影名，并只保留第一个出现的1682
# # def map_movie_names_only(movie_sequence):
# #     # 初始化标记，检查是否已经保留了第一个1682
# #     first_1682_kept = False
# #     filtered_movies = []
# #     for id, rating in movie_sequence:
# #         if id == 1682 and not first_1682_kept:
# #             # 保留第一个1682
# #             filtered_movies.append((id, rating))
# #             first_1682_kept = True
# #         elif id != 1682:
# #             # 保留非1682的电影
# #             filtered_movies.append((id, rating))
# #     # 映射电影ID到名称
# #     return [movie_dict[id] if id in movie_dict else id for (id, rating) in filtered_movies]

# # # 应用这个函数到df的'seq'列来创建只包含电影名称的新列
# # df['movie_names_only'] = df['seq'].apply(map_movie_names_only)

# # # 同时更新'seq'列，使其只包含电影ID，并只保留第一个出现的1682
# # def filter_seq_only(movie_sequence):
# #     first_1682_kept = False
# #     filtered_ids = []
# #     for id, rating in movie_sequence:
# #         if id == 1682 and not first_1682_kept:
# #             filtered_ids.append(id)
# #             first_1682_kept = True
# #         elif id != 1682:
# #             filtered_ids.append(id)
# #     return filtered_ids

# # df['seq_only'] = df['seq'].apply(filter_seq_only)

# # #embeddding
# # import torch
# # from sentence_transformers import SentenceTransformer
# # import numpy as np
# # import pandas as pd

# # from sentence_transformers import SentenceTransformer

# # model = SentenceTransformer(model_name_or_path="/data/projects/wsx/LLaRA/all-MiniLM-L6-v2/")



# # def get_movie_embeddings(movie_list):
# #     embeddings = []
# #     for movies in movie_list:
# #         # 确保每个元素都是字符串类型
# #         movie_string = "。".join(str(movie) for movie in movies)
# #         # 使用模型编码文本
# #         movie_embedding = model.encode(movie_string)
# #         embeddings.append(movie_embedding)
# #     return np.array(embeddings)

# # # 获取电影嵌入向量
# # movie_embeddings = get_movie_embeddings(df['movie_names_only'].tolist())

# # df['movie_embeddings'] = list(movie_embeddings)


# # #按照余弦相似度，找到每个序列最相似的序列
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity
# # embeddings = np.stack(df['movie_embeddings'].values)
# # similarity_matrix = cosine_similarity(embeddings)


# # # 找到每个序列最相似的序列
# # most_similar_indices = np.argmax(similarity_matrix - np.eye(len(similarity_matrix)), axis=1)

# # # 创建一个新列，存储每个序列最相似的序列的索引
# # df['most_similar_seq_index'] = most_similar_indices
# # df['most_similar_seq'] = df['seq_only'].iloc[most_similar_indices].values

# # # 新增添加每个序列最相似的序列的 `next` 值
# # df['most_similar_seq_next'] = df['next'].iloc[most_similar_indices].values


# # output_file_path = '/data/projects/wsx/LLaRA/data/ref/movielens/similar_Test_data.df'
# # df.to_pickle(output_file_path)


import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer

# 全局加载模型
model = SentenceTransformer(model_name_or_path="/data/projects/wsx/LLaRA/all-MiniLM-L6-v2/")

def load_data(file_path):
    return pd.read_pickle(file_path)

def load_movie_dict(item_file):
    item_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', usecols=[0, 1])
    item_df.columns = ['movie_id', 'movie_title']
    movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))
    return movie_dict

def map_movie_names_only(seq, movie_dict):
    return [movie_dict[id] if id in movie_dict else id for (id, rating) in seq]

def extract_sequences(df, movie_dict):
    df['movie_names_only'] = df['seq'].apply(lambda x: map_movie_names_only(x, movie_dict))
    df['seq_only'] = df['seq'].apply(lambda x: [id for (id, rating) in x])
    return df

def get_movie_embeddings(movie_list):
    embeddings = []
    for movies in movie_list:
        movie_string = "。".join(str(movie) for movie in movies)
        movie_embedding = model.encode(movie_string)
        embeddings.append(movie_embedding)
    return np.array(embeddings)

def calculate_similarity(df):
    movie_embeddings = get_movie_embeddings(df['movie_names_only'].tolist())
    df['movie_embeddings'] = list(movie_embeddings)
    embeddings = np.stack(df['movie_embeddings'].values)
    similarity_matrix = cosine_similarity(embeddings)
    most_similar_indices = np.argmax(similarity_matrix - np.eye(len(similarity_matrix)), axis=1)
    df['most_similar_seq_index'] = most_similar_indices
    df['most_similar_seq'] = df['most_similar_seq_index'].apply(lambda idx: df.at[idx, 'seq'])
    return df

def add_most_similar_seq_next(df, movie_dict):
    df['most_similar_seq_next'] = df['next'].iloc[df['most_similar_seq_index']].values
    df['most_similar_seq_name'] = df['most_similar_seq'].apply(lambda x: [movie_dict.get(item[0], "Unknown") for item in x])
    df['most_similar_seq_next_name'] = df['most_similar_seq_next'].apply(lambda x: movie_dict.get(x[0], "Unknown"))
    return df

def save_data(df, output_file_path):
    df.to_pickle(output_file_path)

def process_data(file_path, item_file, output_file_path):
    df = load_data(file_path)
    movie_dict = load_movie_dict(item_file)
    df = extract_sequences(df, movie_dict)
    df = calculate_similarity(df)
    df = add_most_similar_seq_next(df, movie_dict)
    save_data(df, output_file_path)

# 使用函数处理数据
file_path = '/data/projects/wsx/LLaRA/data/ref/movielens/Val_data.df'
item_file = '/data/projects/wsx/LLaRA/data/ref/movielens/u.item'
output_file_path = '/data/projects/wsx/LLaRA/data/ref/movielens/similar_val_data.df'

process_data(file_path, item_file, output_file_path)



##llama-2-7b
import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

import pandas as pd
import random
from torch.utils.data import DataLoader
class MovielensData(data.Dataset):
    def __init__(self, data_dir=r'data/ref/movielens',
                 stage=None,
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=1682
        self.padding_rating=0
        self.check_files()

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        sample = {
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': self.cans_num,
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name'],
            'most_similar_seq': temp['most_similar_seq'],
            'most_similar_seq_next': temp['most_similar_seq_next'],
            'most_similar_seq_name': temp['most_similar_seq_name'],
            'most_similar_seq_next_name': temp['most_similar_seq_next_name'],
            
        }
        return sample
    
    def negative_sampling(self,seq_unpad,next_item):
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

    def check_files(self):
        self.item_id2name=self.get_movie_id2name()
        if self.stage=='train':
            filename="similar_train_data.df"
        elif self.stage=='val':
            filename="similar_val_data.df"
        elif self.stage=='test':
            filename="similar_test_data.df"
        data_path=op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  
    
    def get_mv_title(self,s):
        sub_list=[", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:]+" "+s.replace(sub_s,"")
        return s

    def get_movie_id2name(self):
        movie_id2name = dict()
        item_path=op.join(self.data_dir, 'u.item')
        with open(item_path, 'r', encoding = "ISO-8859-1") as f:
            for l in f.readlines():
                ll = l.strip('\n').split('|')
                movie_id2name[int(ll[0]) - 1] = self.get_mv_title(ll[1][:-7])
        return movie_id2name
    
    def session_data4frame(self, datapath, movie_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove((self.padding_item_id,self.padding_rating))
                except:
                    break
            return x
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        def seq_to_title(x): 
            return [movie_id2name[x_i[0]] for x_i in x]
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        def next_item_title(x): 
            return movie_id2name[x[0]]
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        def get_id_from_tumple(x):
            return x[0]
        def get_id_from_list(x):
            return [i[0] for i in x]
    
        train_data['next'] = train_data['next'].apply(get_id_from_tumple)
        train_data['seq'] = train_data['seq'].apply(get_id_from_list)
        train_data['seq_unpad']=train_data['seq_unpad'].apply(get_id_from_list)
        train_data['most_similar_seq_next'] = train_data['most_similar_seq_next'].apply(get_id_from_tumple)
        train_data['most_similar_seq'] = train_data['most_similar_seq'].apply(get_id_from_list)
   
        return train_data
dataset=MovielensData(data_dir='/data/projects/wsx/LLaRA/data/ref/movielens/',stage='val')
dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
for batch in dataloader:
    print(batch)
    break


import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import LlamaTokenizer, LlamaModel

# 全局加载LLaMA-2-7B模型
model_name_or_path = "/workspace/llama/models_hf/7B"

print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
# 设置pad_token为eos_token
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LlamaModel.from_pretrained(model_name_or_path, local_files_only=True)
model.to(device)
print("Model loaded.")

def load_data(file_path):
    return pd.read_pickle(file_path)

def load_movie_dict(item_file):
    item_df = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', usecols=[0, 1])
    item_df.columns = ['movie_id', 'movie_title']
    movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))
    return movie_dict

def map_movie_names_only(seq, movie_dict):
    return [movie_dict[id] if id in movie_dict else id for (id, rating) in seq]

def extract_sequences(df, movie_dict):
    df['movie_names_only'] = df['seq'].apply(lambda x: map_movie_names_only(x, movie_dict))
    df['seq_only'] = df['seq'].apply(lambda x: [id for (id, rating) in x])
    return df

def get_movie_embeddings(movie_list):
    embeddings = []
    max_length = 512  # 设定一个合理的最大长度
    for movies in movie_list:
        movie_string = " ".join(str(movie) for movie in movies)
        inputs = tokenizer(movie_string, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            movie_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(movie_embedding)
    return np.array(embeddings)

def calculate_similarity(df):
    movie_embeddings = get_movie_embeddings(df['movie_names_only'].tolist())
    df['movie_embeddings'] = list(movie_embeddings)
    embeddings = np.stack(df['movie_embeddings'].values)
    similarity_matrix = cosine_similarity(embeddings)
    most_similar_indices = np.argmax(similarity_matrix - np.eye(len(similarity_matrix)), axis=1)
    df['most_similar_seq_index'] = most_similar_indices
    df['most_similar_seq'] = df['most_similar_seq_index'].apply(lambda idx: df.at[idx, 'seq'])
    return df

def add_most_similar_seq_next(df, movie_dict):
    df['most_similar_seq_next'] = df['next'].iloc[df['most_similar_seq_index']].values
    df['most_similar_seq_name'] = df['most_similar_seq'].apply(lambda x: [movie_dict.get(item[0], "Unknown") for item in x])
    df['most_similar_seq_next_name'] = df['most_similar_seq_next'].apply(lambda x: movie_dict.get(x[0], "Unknown"))
    return df

def save_data(df, output_file_path):
    df.to_pickle(output_file_path)

def process_data(file_path, item_file, output_file_path):
    df = load_data(file_path)
    movie_dict = load_movie_dict(item_file)
    df = extract_sequences(df, movie_dict)
    df = calculate_similarity(df)
    df = add_most_similar_seq_next(df, movie_dict)
    save_data(df, output_file_path)

# 使用函数处理数据
file_path = '/workspace/LLaRA/data/ref/movielens/train_data.df'
item_file = '/workspace/LLaRA/data/ref/movielens/u.item'
output_file_path = '/workspace/LLaRA/data/ref/movielens/similar_val_data.df'

process_data(file_path, item_file, output_file_path)
