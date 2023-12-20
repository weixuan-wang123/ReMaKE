import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_rewrite_quality_zsre, compute_rewrite_quality_counterfact, compute_icl_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel, AutoModelForSequenceClassification, XLMRobertaForSequenceClassification


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, device_map='auto')
                self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'bloomz' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True).cuda()
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            else:
                raise NotImplementedError

            if (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
        else:
            self.model, self.tok = self.model_name


        if hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        
        # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        self.search_model = SentenceTransformer('./model/paraphrase-multilingual-mpnet-base-v2').cuda()
        
        langs = ['cz','de','du','es','fr','pt','ru','th','tr','vi','zh']
        self.memory = {}
        for lang in langs:
            with open(os.path.join("./data/mzsRE/",f"mzsre_test_duplicate_en{lang}.json"), "r", encoding="utf-8") as f:
                lines = json.load(f)
            memory_emb, memory_ans, memory_ques = [], [], []
            for line in lines:
                key = line[lang]['src']
                value = line[lang]['alt']
                memory_ques.append(key)
                memory_ans.append(value)
            self.memory[lang] = {"memory_ques":memory_ques,"memory_ans":memory_ans,"memory_emb":memory_emb}

        with open(os.path.join("./data/mzsRE/",f"mzsre_test_duplicate_enzh.json"), "r", encoding="utf-8") as f:
            lines = json.load(f)
        memory_emb, memory_ans, memory_ques = [], [], []
        for line in lines:
            key = line['en']['src']
            value = line['en']['alt']
            memory_ques.append(key)
            memory_ans.append(value)
        self.memory['en'] = {"memory_ques": memory_ques, "memory_ans": memory_ans, "memory_emb": memory_emb}


        model_name = "./model/XLM-12lang/"

        self.xlmr_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.xlmr_model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")

    def edit(self,
             edited_inputs:  Optional[Dict] = None,
             cross_inputs:  Optional[Dict] = None,
             generalization_inputs:  Optional[Dict] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs:  Optional[Dict] = None,
             keep_original_weight=True,  # 设置为True的时候代表不是序列式地edit，设置为False的时候代表序列式地edit，每次edit完都会改变参数
             lang1="cz",
             lang2="de",
             search="",
             subject=[],
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        # assert source_lang in ["en", "zh"], "source language should in en or zh"

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1


        requests = self._prepare_requests(
            edited_inputs,
            cross_inputs,
            generalization_inputs,
            locality_inputs,
            portability_inputs,
            keep_original_weight,  # 设置为True的时候代表不是序列式地edit，设置为False的时候代表序列式地edit，每次edit完都会改变参数
            lang1,
            lang2,
            search,
            subject,
            **kwargs)



        all_metrics = []
        for i, request in tqdm(enumerate(requests)):
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                metrics = {
                    "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],[''],[''],[''],
                                                     request, self.hparams.device, pre_edit=True, source_lang=lang2)
                }
            else:
                metrics = {
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device)
                }
            all_metrics.append(metrics)


        for i, request in tqdm(enumerate(requests)):
            start = time()

            if self.alg_name == 'IKE':
                prepare_request = request.copy()

                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy = self.model, {}


                icl_examples_cross = self.apply_algo(self.model,self.tok,
                                                     {'search_prompt': prepare_request['cross']['cross']['search_prompt'],
                                                      'search_truth': prepare_request['cross']['cross']['search_truth'],
                                                     'prompt': prepare_request['cross']['cross']['prompt']},
                                                     self.hparams,copy=False, return_orig_weights=True,keep_original_weight=keep_original_weight,train_ds=kwargs['train_ds'],lang=lang1)
                icl_examples_gene = self.apply_algo(self.model, self.tok,
                                                     {'search_prompt': prepare_request['generalization']['rephrase']['search_prompt'],
                                                      'search_truth': prepare_request['generalization']['rephrase']['search_truth'],
                                                      'prompt': prepare_request['generalization']['rephrase']['prompt']},
                                                     self.hparams, copy=False,return_orig_weights=True,keep_original_weight=keep_original_weight,train_ds=kwargs['train_ds'],lang=lang1)
                icl_examples_loca = self.apply_algo(self.model, self.tok,
                                                     {'search_prompt': prepare_request['locality']['neighborhood']['search_prompt'],
                                                      'search_truth': prepare_request['locality']['neighborhood']['search_truth'],
                                                      'prompt': prepare_request['locality']['neighborhood']['prompt']},
                                                     self.hparams, copy=False, return_orig_weights=True,keep_original_weight=keep_original_weight,train_ds=kwargs['train_ds'],lang=lang1)
                icl_examples_port = self.apply_algo(self.model, self.tok,
                                                     {'search_prompt': prepare_request['portability']['one_hop']['search_prompt'],
                                                      'search_truth': prepare_request['portability']['one_hop']['search_truth'],
                                                      'prompt': prepare_request['portability']['one_hop']['prompt']},
                                                     self.hparams, copy=False, return_orig_weights=True,keep_original_weight=keep_original_weight,train_ds=kwargs['train_ds'],lang=lang1)

                exec_time = time() - start
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples_cross,icl_examples_gene,
                                                     icl_examples_loca, icl_examples_port, request, self.hparams.device, source_lang=lang2),
                })

            else:
                prepare_request = request.copy()
                prepare_request["target_new"] = prepare_request["target_new_%s"%source_lang]

                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [prepare_request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start

                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                })
                if self.alg_name == 'KN':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                if 'locality' in all_metrics[i]['post'].keys():
                    for locality_key in request['locality'].keys():
                        assert len(all_metrics[i]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[i]['pre']['locality'][f'{locality_key}_output'])
                        all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = \
                            np.mean(np.equal(all_metrics[i]['post']['locality'][f'{locality_key}_output'],
                                             all_metrics[i]['pre']['locality'][f'{locality_key}_output']))
                        all_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[i]['pre'].pop('locality')


        return all_metrics, edited_model, weights_copy
    


    def batch_edit(self,
                   prompts: List[str],
                   target_new_en: List[str],
                   target_new_zh: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts_en: Optional[List[str]] = None,
                   rephrase_prompts_zh: Optional[List[str]] = None,
                   locality_prompts_en: Optional[List[str]] = None,
                   locality_prompts_zh: Optional[List[str]] = None,
                   locality_ground_truth: Optional[List[str]] = None,
                   keep_original_weight=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new_en)
        assert len(prompts) == len(target_new_zh)

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name) \
               or print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new_en, target_new_zh, ground_truth, rephrase_prompts_en, rephrase_prompts_zh,
                                          locality_prompts_en, locality_prompts_zh, locality_ground_truth, **kwargs)

        assert hasattr(self.hparams, 'batch_size') or \
               print(f'Method {self.alg_name} found, pls specify the batch_size....')

        for record_chunks in self._chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
            )
            exec_time = time() - start


            start = time()
            all_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }

                all_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                all_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)




        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')

        is_singleton = SingletonEditor.is_singleton_method(self.alg_name)



        if is_singleton:
            num_edits = 1 # Single editor method found
        else:
            assert hasattr(self.hparams, 'batch_size') or \
                   print(f'Method {self.alg_name} found, pls set the batch_size correctly')

            num_edits = self.hparams.batch_size

        all_metrics = []

        for record_chunks in tqdm(self._chunks(ds, num_edits), desc='Editing dataset', total=len(ds)/num_edits):

            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight
            )
            exec_time = time() - start

            start = time()
            all_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': request['case_id'],
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }
                all_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                all_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                      self.hparams.device)



        return all_metrics, edited_model, weights_copy




    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]



    def search_memory(self,search,index,question,answer,lang):
        if search == 'classifier':
            flag = 0
            num = index
            text = [en + self.xlmr_tokenizer.sep_token + question for en in self.memory[lang]['memory_ques']]
            input_ids = self.xlmr_tokenizer(text, return_tensors='pt', padding=True).to('cuda')
            output = self.xlmr_model(**input_ids)
            big_val, big_idx = torch.max(output.logits.log_softmax(-1), dim=1)
            if 0 in big_idx:
                indices = np.where(big_idx.detach().cpu().numpy() == 0)[0]
                max_index = indices[np.argmax(big_val.detach().cpu().numpy()[indices])]
                m_q, m_a = self.memory[lang]['memory_ques'][max_index], self.memory[lang]['memory_ans'][max_index]
                if max_index == index:
                    flag = 1
            else:
                m_q, m_a = "", ""
            return m_q, m_a, flag

    def _prepare_requests(self,
                          edited_inputs: Optional[Dict] = None,
                          cross_inputs: Optional[Dict] = None,
                          generalization_inputs: Optional[Dict] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          keep_original_weight=True,
                          lang1="cz",
                          lang2="de",
                          search="",
                          subject=[],
                          **kwargs
                          ):

        requests = [{
            "edited": {},
            "cross": {},
            "generalization": {},
            "locality": {},
            "portability": {},
            "subject": sub
        } for sub in subject]



        for key in edited_inputs.keys():
            for i, request in enumerate(requests):
                request['edited'].update(
                    {
                        key: {
                            f'prompt': edited_inputs[key]['prompt'][i],
                            f'ground_truth': edited_inputs[key]['ground_truth'][i]
                        }
                    }
                )

        for key in cross_inputs.keys():
            cross_cnt = 0
            for i, request in enumerate(requests):
                if search == "":
                    request['cross'].update(
                        {
                            key: {
                                f'prompt': cross_inputs[key]['prompt'][i],
                                f'ground_truth': cross_inputs[key]['ground_truth'][i],
                                f'search_prompt': edited_inputs['edited_english']['prompt'][i],
                                f'search_truth': edited_inputs['edited_english']['ground_truth'][i]
                            }
                        }
                    )
                else:
                    search_prompt, search_truth,flag = self.search_memory(search, i,cross_inputs[key]['prompt'][i],cross_inputs[key]['ground_truth'][i],lang1)
                    request['cross'].update(
                        {
                            key: {
                                f'prompt': cross_inputs[key]['prompt'][i],
                                f'ground_truth': cross_inputs[key]['ground_truth'][i],
                                f'search_prompt': search_prompt,
                                f'search_truth': search_truth
                            }
                        }
                    )
                    cross_cnt += flag
            print("{0}2{1} %%%%%%%%%%%%%%cross search ratio%%%%%%%%%%%%%% {2}".format(lang1,lang2,cross_cnt/len(requests)))

        for key in generalization_inputs.keys():
            gene_cnt = 0
            for i, request in enumerate(requests):
                if search == "":
                    request['generalization'].update(
                        {
                            key: {
                                f'prompt': generalization_inputs[key]['prompt'][i],
                                f'ground_truth': generalization_inputs[key]['ground_truth'][i],
                                f'search_prompt': edited_inputs['edited_english']['prompt'][i],
                                f'search_truth': edited_inputs['edited_english']['ground_truth'][i]
                            }
                        }
                    )
                else:
                    search_prompt, search_truth, flag = self.search_memory(search, i, generalization_inputs[key]['prompt'][i],generalization_inputs[key]['ground_truth'][i],lang1)
                    request['generalization'].update(
                        {
                            key: {
                                f'prompt': generalization_inputs[key]['prompt'][i],
                                f'ground_truth': generalization_inputs[key]['ground_truth'][i],
                                f'search_prompt': search_prompt,
                                f'search_truth': search_truth
                            }
                        }
                    )
                    gene_cnt += flag
            print("{0}2{1} %%%%%%%%%%%%%%gene search ratio%%%%%%%%%%%%%% {2}".format(lang1,lang2,gene_cnt / len(requests)))

        for key in locality_inputs.keys():
            loca_cnt = 0
            for i, request in enumerate(requests):
                if search == "":
                    request['locality'].update(
                        {
                            key: {
                                f'prompt': locality_inputs[key]['prompt'][i],
                                f'ground_truth': locality_inputs[key]['ground_truth'][i],
                                f'search_prompt': edited_inputs['edited_english']['prompt'][i],
                                f'search_truth': edited_inputs['edited_english']['ground_truth'][i]
                            }
                        }
                    )
                else:
                    search_prompt, search_truth, flag = self.search_memory(search, i,locality_inputs[key]['prompt'][i],locality_inputs[key]['ground_truth'][i],lang1)
                    request['locality'].update(
                        {
                            key: {
                                f'prompt': locality_inputs[key]['prompt'][i],
                                f'ground_truth': locality_inputs[key]['ground_truth'][i],
                                f'search_prompt': search_prompt,
                                f'search_truth': search_truth
                            }
                        }
                    )
                    loca_cnt += flag
            print("{0}2{1} %%%%%%%%%%%%%%loca search ratio%%%%%%%%%%%%%% {2}".format(lang1,lang2,loca_cnt / len(requests)))

        for key in portability_inputs.keys():
            port_cnt = 0
            for i, request in enumerate(requests):
                if search == "":
                    request['portability'].update(
                        {
                            key: {
                                f'prompt': portability_inputs[key]['prompt'][i],
                                f'ground_truth': portability_inputs[key]['ground_truth'][i],
                                f'search_prompt': edited_inputs['edited_english']['prompt'][i],
                                f'search_truth': edited_inputs['edited_english']['ground_truth'][i]
                            }
                        }
                    )
                else:
                    search_prompt, search_truth, flag = self.search_memory(search, i,portability_inputs[key]['prompt'][i],portability_inputs[key]['ground_truth'][i],lang1)
                    request['portability'].update(
                        {
                            key: {
                                f'prompt': portability_inputs[key]['prompt'][i],
                                f'ground_truth': portability_inputs[key]['ground_truth'][i],
                                f'search_prompt': search_prompt,
                                f'search_truth': search_truth
                            }
                        }
                    )
                    port_cnt += flag
            print("{0}2{1} %%%%%%%%%%%%%%port search ratio%%%%%%%%%%%%%% {2}".format(lang1,lang2,port_cnt / len(requests)))





        return requests





