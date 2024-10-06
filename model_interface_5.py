import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
import json
from unsloth import FastLanguageModel 
import re
import copy
from transformers import Trainer, BitsAndBytesConfig, AutoTokenizer
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import os
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
import math

score_instruct = """You are a system that recommends movies based on viewing history. Please evaluate the similarity between each watch history in the candidate list and the single target watch history. Rate the similarity on a scale from 1 to 10 between , where 1 is not similar at all and 10 is very similar.\n
Please output the similarity ratings in JSON format. Here is the format:
["Watch History 1": score, "Watch History 2": score, "Watch History 3": score, "Watch History 4": score, "Watch History 5": score, "Watch History 6": score, "Watch History 7": score, "Watch History 8": score, "Watch History 9": score, "Watch History 10": score]\n"""

score_history = """Candidate Watch History:
{MOVIE_LISTS} \n
Target Watch History:
{TARGET_MOVIE} \n
Please output the similarity ratings in JSON format. The output should only contain the JSON object with similarity scores, without any additional text. Output:"""

reco_instruct = """You are a movie recommendation system. Below are some similar users' viewing histories and the next movie they are likely to choose. Based on the current user's viewing history, your task is to recommend the next movie for this user. Instructions: 1. Recommend one movie title. 2。It **must** be from the candidate pool only. 
Please output the recommendation in the format below:
['Recommendation': movie_name] \n
"""

reco_prompt_history = """Similar user {i}: He/She has watched {SimilarHistory}. Based on this, She/he chose {SimilarChoice} to watch next. \n"""

reco_prompt_instruct = """The visit history of this user is: {HistoryHere}. Recommend one movie from the following set of titles: {CansHere}. Output:"""

# TERMINATOR="\n"

class MInterface(pl.LightningModule):
    def __init__(self, 
                 **kargs):
        super().__init__()
        # self.adapter_path = "/mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model_adapter"
        self.adapter_path = kargs["adapter_path"]
        # self.score_model_path = "/mnt/bn/data-tns-live-llm/leon/datasets/rec/score_model"
        self.score_model_path = kargs["score_model_path"]
        self.output_dir = kargs["output_dir"]
        self.save_hyperparameters()
        self.model_max_length = kargs["model_max_length"]
        self.is_test = (kargs["mode"]=="test")
        print(f"is test {self.is_test}")
        self.unsloth = kargs["unsloth"]
        if not self.unsloth: 
            self.get_quant_model(self.hparams.llm_path)
            if not self.is_test: self.get_score_model()
            else:
                print("Loading encoder")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else: 
            self.load_unsloth_llm(self.hparams.llm_path)
            if self.is_test: 
                print("Loading encoder")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.batch_size = kargs["batch_size"]
        # self.load_rec_model(self.hparams.rec_model_path)
        # self.load_projector()
    
    def get_score_model(self):
        self.score_model = LlamaForCausalLM.from_pretrained(self.score_model_path, load_in_4bit=True)

    def copy_and_quantization(self, epoch, batch_idx):
        self.llama_model.save_pretrained(f"{self.adapter_path}/{epoch}_{batch_idx}")
        self.score_model = LlamaForCausalLM.from_pretrained(self.hparams.llm_path)
        self.score_model.resize_token_embeddings(len(self.llama_tokenizer))
        self.score_model = PeftModel.from_pretrained(self.score_model, f"{self.adapter_path}/{epoch}_{batch_idx}")
        self.score_model = self.score_model.merge_and_unload()
        self.score_model.save_pretrained(self.score_model_path)
        self.score_model = LlamaForCausalLM.from_pretrained(self.score_model_path, load_in_4bit=True)
    
    # 对相似 demo 进行打分 TODO
    def score_demo(self, input):
        movie_list = [" ".join(names) for names in input["most_similar_seq_name"]]
        movie_lists = ""
        for i, name in enumerate(movie_list):
            movie_lists += f"Watch History {i+1}: {name} \n"
        target_movie = " ".join(input["seq_name"][:input["len_seq"]])
        # print(movie_lists)
        input_prompt = score_instruct + score_history.format_map({"MOVIE_LISTS":movie_lists,"TARGET_MOVIE":target_movie})
        # with open("/mnt/bn/data-tns-live-llm/leon/LLaRA-similar_seq_as_demo-/tmp.txt","a") as f:
        #     f.write(input_prompt)
        input = self.llama_tokenizer(input_prompt, return_tensors="pt")

        # effective_token_lengths = torch.sum(input["attention_mask"], dim=1)
        # average_effective_token_length = torch.mean(effective_token_lengths.float()).item()
        # print(average_effective_token_length)

        output = self.score_model.generate(input["input_ids"].cuda(), temperature=0.1, max_new_tokens=128, repetition_penalty=1.1).cpu()[0]
        org_output = self.llama_tokenizer.decode(output)
        try:
            output = "["+org_output.split("Output:")[1].split("[")[1].split("]")[0]+"]"
            output = output.strip("[]")
            # 提取键值对，并处理键中的单引号
            matches = re.findall(r"'(.*?)'\s*:\s*([\d.]+)", output)
            # 转换为字典，数值转换为浮点数
            output = {key.replace("\\'", "'"): float(value) for key, value in matches}
            # print(output)
            data_list = list(output.items())[:10]
            # 按评分进行排序（从高到低）。由于值都是浮点数，所以直接用sorted排序。
            sorted_data_list = sorted(data_list, key=lambda x: x[1], reverse=True)
            # 提取排序后前 5 个评分对应的索引
            top_5_idx = [data_list.index(item) for item in sorted_data_list[:5]]

            # output = json.loads(output)
            # scores = re.findall(r':\s*([\d.]+)', output)[:10]
            # scores = np.array([-int(score) for score in scores])
            # print(scores)
            # top_5_idx = np.argsort(scores)[:5]

            # 获取分数最高的前5个键
            # sorted_items = sorted(output.items(), key=lambda item: item[1], reverse=True)
            # top_5_idx = [int(re.findall(r'\d+', item[0])[0])-1 for item in sorted_items[:5]]
            return top_5_idx
        except Exception as e:
            print(e)
            # print("bad format, cant decode")
        return random.sample(range(10),5)
    
    # tokenize
    def format_fn(self, input):
        if not self.unsloth:
            top_5_idx = self.score_demo(input)
            similar_historys = [input["most_similar_seq_name"][idx] for idx in top_5_idx]
            similar_choices = [input["most_similar_seq_next_name"][idx] for idx in top_5_idx]
        else:
            similar_historys = input["most_similar_seq_name"][:5]
            similar_choices = input["most_similar_seq_next_name"][:5]
        
        # def format_similar_history(input):
        #     return ", ".join([seq_title + ' [SimilarHistoryEmb]' for seq_title in input])
            
        # def format_similar_choice(input):
        #     return input + ' [SimilarChoiceEmb]'
        
        # # 添加 Emb token
        # similar_historys = [format_similar_history(similar_history) for similar_history in similar_historys]
        # similar_choices = [format_similar_choice(similar_choice) for similar_choice in similar_choices]
        # history_here = ", ".join([seq_title+' [HistoryEmb]' for seq_title in input['seq_name']])
        # cans_here = ", ".join([can_title+' [CansEmb]' for can_title in input['cans_name']])

        demos = [reco_prompt_history.format_map({"i":i,"SimilarHistory":similar_history, "SimilarChoice":similar_choice}) for i, (similar_history,similar_choice) in enumerate(zip(similar_historys, similar_choices))]
        demos = "".join(demos)
        instruction = reco_prompt_instruct.format_map({"HistoryHere":input["seq_name"], "CansHere":input["cans_name"]})
        # instruction = reco_prompt_instruct.format_map({"HistoryHere":history_here, "CansHere":cans_here})
        instruction = reco_instruct + demos+" "+instruction
        # print("instruction", instruction)
        return instruction

    def collate_fn(self, batch):
        # print("batch ", batch)
        # print("batch most similar seq",[sample["most_similar_seq"] for sample in batch])
        # print("batch most similar seq next",[sample["most_similar_seq_next"] for sample in batch])
        cans_name = [input["cans_name"] for input in batch]
        inputs_text = [self.format_fn(input) for input in batch]
        targets_text = ["['Recommendation': {}]{}".format(input['correct_answer'], self.llama_tokenizer.eos_token) for input in batch]

        if self.llama_model.training:
            # print(targets_text)
            # targets_text = [target_text + TERMINATOR for target_text in targets_text]
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_text)]

            batch_tokens = self.llama_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)

            # effective_token_lengths = torch.sum(batch_tokens["attention_mask"], dim=1)
            # average_effective_token_length = torch.mean(effective_token_lengths.float()).item()
            # print(average_effective_token_length)

            # most_similar_seq_next=[sample['most_similar_seq_next'] for sample in batch]
            with open("./log.txt","a") as f:
                f.write(json.dumps(f"inputs pair: {inputs_pair} \n"))
 
            new_batch = {
                "tokens": batch_tokens,
                "seq": torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                "cans": torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                "len_seq": torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                "len_cans": torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                "most_similar_seq": torch.stack([torch.tensor(sample['most_similar_seq']) for sample in batch], dim=0),
                "most_similar_seq_next": torch.stack([torch.tensor(sample['most_similar_seq_next']) for sample in batch], dim=0)
            }
        else:
            # print(inputs_text)
            batch_tokens = self.llama_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            cans_name = [sample['cans_name'] for sample in batch]

            # effective_token_lengths = torch.sum(batch_tokens["attention_mask"], dim=1)
            # average_effective_token_length = torch.mean(effective_token_lengths.float()).item()
            # print(average_effective_token_length)

            # most_similar_seq_next=[sample['most_similar_seq_next'] for sample in batch]
            new_batch = {
                "tokens": batch_tokens,
                "seq": torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                "cans": torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                "len_seq": torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                "len_cans": torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                "correct_answer": targets_text,
                "cans_name": cans_name,
                "most_similar_seq": torch.stack([torch.tensor(sample['most_similar_seq']) for sample in batch], dim=0),
                "most_similar_seq_next": torch.stack([torch.tensor(sample['most_similar_seq_next']) for sample in batch], dim=0)
            }
        
        # tmp1 = new_batch['most_similar_seq'].shape
        # tmp2 = new_batch['most_similar_seq_next'].shape
        # print(f"most_similar_seq {tmp1} most_similar_seq_next {tmp2}")
        return new_batch
    
    def batch_preprocess(self, batch):
        batch = self.collate_fn(batch)
        batch["tokens"].input_ids = batch["tokens"].input_ids.cuda()
        batch["tokens"].attention_mask = batch["tokens"].attention_mask.cuda()
        if self.llama_model.training:
            batch["tokens"].token_type_ids = batch["tokens"].token_type_ids.cuda()
        return batch

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        # print("targets", targets.shape)
        # tmp = (batch["tokens"].token_type_ids == 0)[:,1:]
        # print("mask", tmp.shape)
        # print("token_type_ids",batch["tokens"].token_type_ids)
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0), -100)
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def generate(self, batch, temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1):
        # input_embeds = self.wrap_emb(batch)
        # Convert input_embeds to float32
        # input_embeds = input_embeds.to(torch.float32)
        generate_ids = self.llama_model.generate(
            input_ids = batch["tokens"].input_ids,
            # inputs_embeds=input_embeds, 
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            )
        output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        batch=self.batch_preprocess(batch)
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        # if batch["flag"]:
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = False
        # else:
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = True
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        #每1000step 更新 score model
        # print(batch_idx)
        if (batch_idx+1)%1000==0: 
            self.copy_and_quantization(self.current_epoch, batch_idx)
            # if not self.unsloth: self.copy_and_quantization(self.current_epoch)
            # else: 
            #     self.llama_model.save_pretrained(self.output_dir)
            #     self.llama_tokenizer.save_pretrained(self.output_dir)
        return loss
    
    def on_validation_epoch_start(self):
        if self.unsloth: FastLanguageModel.for_inference(self.llama_model)
        self.val_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch=self.batch_preprocess(batch)
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        if self.unsloth: FastLanguageModel.for_training(self.llama_model)
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.val_content)
        metric=hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    def on_test_epoch_start(self):
        self.test_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch=self.batch_preprocess(batch)
        generate_output = self.generate(batch)
        output=[]
        for i, generate in enumerate(generate_output):
            # if i == 0: print(generate)
            try: 
                generate = generate.split("Output:")[1].split("['Recommendation': ")[1]
                # if len(generate)>2: generate = "".join(generate[1:])
                # else: generate = generate[1]
                generate = generate.split("]")[0].strip()
                # print(generate)
            except: 
                print("generation in bad format")
                print(generate_output[i])

            # real = batch['correct_answer'][i].split(": ")
            # if len(real)>2: real = "".join(real[1:])
            # else: real = real[1]
            # real = real.split("]")[0].strip()
            real = batch['correct_answer'][i].split("['Recommendation': ")[1].split("]")[0].strip()
            
            # real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            # generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        # metric=hr*prediction_valid_ratio
        ndcg_score_1, retrieved_cans_1 = self.calculate_dcg_at_n(self.test_content,1)
        ndcg_score_5, retrieved_cans_5 = self.calculate_dcg_at_n(self.test_content,5)
        ndcg_score_10, retrieved_cans_10 = self.calculate_dcg_at_n(self.test_content,10)
        ndcg_score_20, retrieved_cans_20 = self.calculate_dcg_at_n(self.test_content,20)
        hr_1=self.calculate_hr1(self.test_content, retrieved_cans_1)
        hr_5=self.calculate_hr1(self.test_content, retrieved_cans_5)
        hr_10=self.calculate_hr1(self.test_content, retrieved_cans_10)
        hr_20=self.calculate_hr1(self.test_content, retrieved_cans_20)
        self.log("DCG@1", ndcg_score_1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("DCG@5", ndcg_score_5, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("DCG@10", ndcg_score_10, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("DCG@20", ndcg_score_20, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        # self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('hr@1', hr_1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('hr@5', hr_5, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('hr@10', hr_10, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('hr@20', hr_20, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        # self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            # {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        ])

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                  max_step=max_step,
                                                  min_lr=self.hparams.lr_decay_min_lr,
                                                  init_lr=self.hparams.lr,
                                                  warmup_steps=warmup_steps,
                                                  warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")
        
    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass
    
    def load_unsloth_llm(self, llm_path):
        max_seq_length = 2048
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        dtype=None

        self.llama_model, self.llama_tokenizer = FastLanguageModel.from_pretrained(
            model_name = "/mnt/bn/data-tns-live-llm/leon/datasets/llama-3-8b-bnb-4bit/",
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = True,
        )

        self.llama_tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '[PH]',
                '[HistoryEmb]',
                '[CansEmb]',
                '[ItemEmb]',
                '[SimilarHistoryEmb]',
                '[SimilarChoiceEmb]'
            ]
        })
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.is_test: self.llama_model = PeftModel.from_pretrained(self.llama_model, llm_path)

        if not self.is_test:
            print("loading lora parameters")
            self.llama_model = FastLanguageModel.get_peft_model(
                self.llama_model,
                r = 16,
                target_modules = ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                lora_alpha = 32,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                max_seq_length = max_seq_length,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
        else:
            FastLanguageModel.for_inference(self.llama_model)
    
    def get_quant_model(self, llm_path):
        if not self.is_test:
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.llama_tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/data-tns-live-llm/leon/datasets/Llama-2-7b-hf", model_max_length=self.model_max_length)
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.padding_side="left"
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llama_tokenizer.add_special_tokens({
                'additional_special_tokens': [
                    '[PH]',
                    '[HistoryEmb]',
                    '[CansEmb]',
                    '[ItemEmb]',
                    '[SimilarHistoryEmb]',
                    '[SimilarChoiceEmb]'
                ]
            })

            self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config)
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            self.llama_model.generation_config.pad_token_id = self.llama_tokenizer.eos_token_id

            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            self.llama_tokenizer.save_pretrained(self.score_model_path)
        else:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_path, model_max_length=self.model_max_length)
            self.llama_tokenizer.padding_side="left"
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(self.llama_tokenizer.pad_token)
            self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
            self.llama_model = PeftModel.from_pretrained(self.llama_model, self.adapter_path, is_trainable=False)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            self.score_model = self.llama_model
        
    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "left"
        # self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        self.llama_tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '[PH]',
                '[HistoryEmb]',
                '[CansEmb]',
                '[ItemEmb]',
                '[SimilarHistoryEmb]',
                '[SimilarChoiceEmb]'
            ]
        })
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()
 
        print('Loading LLAMA Done')
    
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        # input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids).detach()
        return input_embeds
        # his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        # cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        # item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        # his_item_embeds= self.encode_items(batch["seq"])
        # cans_item_embeds= self.encode_items(batch["cans"])
        # item_embeds=self.encode_items(batch["item_id"])

        # 获取所有特殊标记的 token_id
        his_token_id = self.llama_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        cans_token_id = self.llama_tokenizer("[CansEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        item_token_id = self.llama_tokenizer("[ItemEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        similar_history_token_id = self.llama_tokenizer("[SimilarHistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        similar_choice_token_id = self.llama_tokenizer("[SimilarChoiceEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()

        # 编码不同的 embeddings
        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        item_embeds = self.encode_items(batch["item_id"])
        similar_history_embeds = self.encode_items(batch["most_similar_seq"])
        # print(f"similar_history_embeds {similar_history_embeds.shape}")
        similar_choice_embeds = self.encode_items(batch["most_similar_seq_next"])
        # print(f"similar_choice_embeds {similar_choice_embeds.shape}")
                
        # for i in range(len(batch["len_seq"])):
        #     if (batch["tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
        #         idx_tensor=(batch["tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
        #         for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
        #             input_embeds[i,idx]=item_emb
        #     if (batch["tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
        #         idx_tensor=(batch["tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
        #         for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
        #             input_embeds[i,idx]=item_emb
        #     if (batch["tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
        #         idx=(batch["tokens"].input_ids[i]==item_token_id).nonzero().item()
        #         input_embeds[i,idx]=item_embeds[i]
                
        for i in range(len(batch["len_seq"])):
            # 处理 [HistoryEmb] 标记
            if (batch["tokens"].input_ids[i] == his_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, :batch["len_seq"][i].item()]):
                    input_embeds[i, idx] = item_emb
            
            # 处理 [CansEmb] 标记
            if (batch["tokens"].input_ids[i] == cans_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, cans_item_embeds[i, :batch["len_cans"][i].item()]):
                    input_embeds[i, idx] = item_emb
            
            # 处理 [ItemEmb] 标记
            if (batch["tokens"].input_ids[i] == item_token_id).nonzero().shape[0] > 0:
                idx = (batch["tokens"].input_ids[i] == item_token_id).nonzero().item()
                input_embeds[i, idx] = item_embeds[i]
            
            # 处理 [SimilarHistoryEmb] 标记
            if (batch["tokens"].input_ids[i] == similar_history_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == similar_history_token_id).nonzero().view(5,-1) #只有5个similar history
                # print(f"idx_tensor {idx_tensor} \n shape {idx_tensor.shape}")
                tmp=batch["len_seq"][i].item()
                # print(f"len_seq {tmp}")
                # print(f"similar_history_embeds {similar_history_embeds.shape}")
                for idxs, item_embs in zip(idx_tensor, similar_history_embeds[i, :5]): #这里看是否进行打分排序，如果不打分就直接选择前5 个
                    for idx, item_emb in zip(idxs, item_embs):
                        tmp=input_embeds[i, idx]
                        # print(f"input_embeds {tmp.shape}")
                        # print(f"item_emb {item_emb.shape}")
                        input_embeds[i, idx] = item_emb
                    # input_embeds[i, idx] = item_emb
            
            # 处理 [SimilarChoiceEmb] 标记
            if (batch["tokens"].input_ids[i] == similar_choice_token_id).nonzero().shape[0] > 0:
                idxs = (batch["tokens"].input_ids[i] == similar_choice_token_id).nonzero().view(-1)
                for idx, similar_choice_embed in zip(idxs,similar_choice_embeds[i][:5]):
                    input_embeds[i, idx] = similar_choice_embed
    
        return input_embeds

    def calculate_ndcg1(self, eval_content):
        correct_num = 0
        total_num = 0
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = generate.strip().lower().strip()
            real = real.strip().lower().strip()
            cans = [item.strip().lower().strip() for item in cans]
            
            # 假设只有一个候选生成（cans），那么 NDCG@1 就是正确答案是否是这个候选
            if generate == real:
                correct_num += 1
        
        ndcg1 = correct_num / total_num if total_num > 0 else 0
        return ndcg1
    
    def retrieve_candidates(self, query, corpus, top_k):
        """
        使用encoder检索最相似的N个候选句子。
        :param query: 查询句子
        :param corpus: 候选句子集合
        :param top_k: 检索的候选数量
        :return: 检索到的候选句子及其相似度分数
        """
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        corpus_embeddings = self.encoder.encode(corpus, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
        top_indices = similarities[0].topk(top_k, largest=True)[1].tolist()
        top_similarities = similarities[0][top_indices]
        return [corpus[i] for i in top_indices], top_similarities

    def calculate_dcg(self, scores, k):
        """
        计算DCG@k。
        :param scores: 相关性分数列表
        :param k: 计算DCG@k
        :return: DCG@k值
        """
        dcg = 0
        for i in range(min(k, len(scores))):
            dcg += scores[i] / math.log2(i + 2)
        return dcg

    def calculate_dcg_at_n(self, eval_content, N=3):
        print(f"DCG N: {N}")
        ndcg_num = 0
        total_num = 0
        retrieved_candidates = []
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = generate.strip().lower().strip()
            real = real.strip().lower().strip()
            cans = [item.strip().lower().strip() for item in cans]

            # 检索逻辑
            retrieved_cans, similarities = self.retrieve_candidates(generate, cans, N)

            # 计算NDCG@N
            ndcg_at_n = self.calculate_dcg(similarities.cpu().numpy(), N)
            ndcg_num += ndcg_at_n
            # print(retrieved_cans[:N])
            retrieved_candidates.append(retrieved_cans[:N])

        ndcg_score = ndcg_num / total_num
        return ndcg_score, retrieved_candidates




    # def load_projector(self):
    #     name = self.hparams.model_name
    #     camel_name = ''.join([i.capitalize() for i in name.split('_')])
    #     try:
    #         Model = getattr(importlib.import_module(
    #             '.'+name, package=__package__), camel_name)
    #     except:
    #         raise ValueError(
    #             f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
    #     self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)

    # def instancialize(self, Model, **other_args):
    #     class_args = inspect.getargspec(Model.__init__).args[1:]
    #     inkeys = self.hparams.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = getattr(self.hparams, arg)
    #     args1.update(other_args)
    #     return Model(**args1)

    # def load_rec_model(self, rec_model_path):
    #     print('Loading Rec Model')
    #     self.rec_model = torch.load(rec_model_path, map_location="cpu")
    #     self.rec_model.eval()
    #     for name, param in self.rec_model.named_parameters():
    #         param.requires_grad = False
    #     print('Loding Rec model Done')

    # def encode_items(self, seq):
    #     seq = seq.cuda()
    #     if self.hparams.rec_embed=="SASRec":
    #         item_rec_embs=self.rec_model.cacu_x(seq)
    #     elif self.hparams.rec_embed in ['Caser','GRU']:
    #         item_rec_embs=self.rec_model.item_embeddings(seq)
    #     item_txt_embs=self.projector(item_rec_embs)
    #     return item_txt_embs

    def calculate_hr1(self, eval_content, retrieved_content):
        correct_num = 0
        total_num = len(eval_content["real"])
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            real = real.strip().lower().strip()
            if real in retrieved_content[i]: 
                correct_num+=1
            else: 
                cans = eval_content["cans"][i]
                # print(f"retrieved_content: {retrieved_content[i]} \n cans: {cans} \n real: {real}")
            
        hr1 = correct_num / total_num
        # for i, generate in enumerate(eval_content["generate"]):
        #     # print(f"Debug: generate type: {type(generate)}")
        #     # print("generate:",generate)
        #     real = eval_content["real"][i]
        #     cans = eval_content["cans"][i]
        #     total_num += 1
        #     generate = generate.strip().lower().strip()
        #     real = real.strip().lower().strip()
        #     cans = [item.strip().lower().strip() for item in cans]
        #     gen_cans_list = []
        #     for cans_item in cans:
        #         if cans_item in generate:
        #             gen_cans_list.append(cans_item)
        #     if len(gen_cans_list) == 1:
        #         valid_num += 1
        #         if real == gen_cans_list[0]:
        #             correct_num += 1
        # valid_ratio = valid_num / total_num
        # if valid_num > 0:
        #     hr1 = correct_num / valid_num
        # else:
        #     hr1 = 0
        return hr1
