"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from typing import List

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality


def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples_cross,
    icl_examples_gene,
    icl_examples_loca,
    icl_examples_port,
    record: typing.Dict,
    device,
    pre_edit: bool = False,
    source_lang: str = "en",
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    search_prompt, search_truth = record['cross']['cross']['search_prompt'], record['cross']['cross']['search_truth']
    cross_prompt, cross_truth = record['cross']['cross']['prompt'], record['cross']['cross']['ground_truth']

    if pre_edit:
        edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, [''], cross_truth,f'Question: {cross_prompt} Answer:')
    else:

        if search_prompt != "":
            new_fact = f'Question: {search_prompt} Answer: {search_truth}\nQuestion: {cross_prompt} Answer:'

        else:
            new_fact = f'Question: {cross_prompt} Answer:'
        if len(icl_examples_cross) > 0:
            print(icl_examples_cross[-1] + new_fact)
        edit_acc_ans, edit_acc_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_cross, cross_truth, new_fact)


    ret = {
        f"reliability": {
            "ans": edit_acc_ans,
            "target": edit_acc_target
        }
    }
    ret['generalization'] = {}
    ret['locality'] = {}
    ret['portability'] = {}

    for key in record['generalization'].keys():
        search_prompt, search_truth = record['generalization'][key]['search_prompt'], record['generalization'][key]['search_truth']
        if pre_edit:
            gene_ans, gene_target = icl_lm_eval(model, model_name, hparams, tok, [''],record['generalization'][key]['ground_truth'],f"Question: {record['generalization'][key]['prompt']} Answer:")
        else:
            if search_prompt != "":
                
                gene_ans, gene_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_gene,
                                                    record['generalization'][key]['ground_truth'],
                                                    f"Question: {search_prompt} Answer: {search_truth}\nQuestion: {record['generalization'][key]['prompt']} Answer:")
            else:
                gene_ans, gene_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_gene,
                                                    record['generalization'][key]['ground_truth'],
                                                    f"Question: {record['generalization'][key]['prompt']} Answer:")

        ret['generalization'][f'{key}_acc'] = {
            "ans": gene_ans,
            "target": gene_target
        }


    for key in record['locality'].keys():
        search_prompt, search_truth = record['locality'][key]['search_prompt'], record['locality'][key]['search_truth']
        if pre_edit:
            loca_ans, loca_target = icl_lm_eval(model, model_name, hparams, tok, [''],record['locality'][key]['ground_truth'],f"Question: {record['locality'][key]['prompt']} Answer:")
        else:
            if search_prompt != "":
                
                loca_ans, loca_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_loca,
                                                    record['locality'][key]['ground_truth'],
                                                    f"Question: {search_prompt} Answer: {search_truth}\nQuestion: {record['locality'][key]['prompt']} Answer:")
            else:
                loca_ans, loca_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_loca,
                                                    record['locality'][key]['ground_truth'],
                                                    f"Question: {record['locality'][key]['prompt']} Answer:")

        ret['locality'][f'{key}_acc'] = {
            "ans": loca_ans,
            "target": loca_target
        }

        for key in record['portability'].keys():
            search_prompt, search_truth = record['portability'][key]['search_prompt'], record['portability'][key]['search_truth']
            if pre_edit:
                gene_ans, gene_target = icl_lm_eval(model, model_name, hparams, tok, [''],record['portability'][key]['ground_truth'],
                                                    f"Question: {record['portability'][key]['prompt']} Answer:")
            else:
                if search_prompt != "":

                    gene_ans, gene_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_port,
                                                        record['portability'][key]['ground_truth'],
                                                        f"Question: {search_prompt} Answer: {search_truth}\nQuestion: {record['portability'][key]['prompt']} Answer:")
                else:
                    gene_ans, gene_target = icl_lm_eval(model, model_name, hparams, tok, icl_examples_port,
                                                        record['portability'][key]['ground_truth'],
                                                        f"Question: {record['portability'][key]['prompt']} Answer:")

            ret['portability'][f'{key}_acc'] = {
                "ans": gene_ans,
                "target": gene_target
            }

    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cudAnswer:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower() or 'baichuan' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt',max_length=1520)
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]

        
        ans_idss = ans.detach().cpu().numpy().tolist()
        target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
        if not isinstance(ans_idss, list):
            ans_idss = [ans_idss]

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
        textual_target = tokenizer.decode(target_idss, skip_special_tokens=True)

        if neighborhood:
            return textual_ans
        return textual_ans, textual_target
    else:
        target_ids = tokenizer(target+'</s>', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()
        target_idss = target_ids.detach().cpu().squeeze().numpy().tolist()
        if not isinstance(ans_idss, list):
            ans_idss = [ans_idss]

        textual_ans = tokenizer.decode(ans_idss, skip_special_tokens=True)
        textual_target = tokenizer.decode(target_idss, skip_special_tokens=True)

        if neighborhood:
            return textual_ans

        return textual_ans.strip(), textual_target.strip()
        

# TODO: Support GPT Evaluation(predict token one by one)
def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    lang: str = "en",
) -> typing.Dict:

    if 't5' in model_name.lower():
        stuff_probs = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                        prompt,
                                                        target_new,
                                                        device)
    elif 'gpt' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # inp_targets = [
        #     tok.decode(target_tok[i])
        #     for i in range(len(target_tok))
        # ]
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'llama' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] #erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'baichuan' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"] #erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)


    # Structure the restuls as a dictionary.

    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    
    if not test_rephrase:
        ret = {
            f"{key}_acc": {
                "ans": textual_ans,
                "target": textual_target
            }
        }
    else:
        ret = {
            f"{key}_acc_{lang}": {
                "ans": textual_ans,
                "target": textual_target
            }
        }

    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
    lang: str = "en",
) -> typing.Dict:

    if 't5' in model_name.lower():
        locality_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                                 prompt,
                                                                 locality_ground_truth,
                                                                 device,
                                                                 locality=True)
    elif 'gpt' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])

        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    elif 'llama' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    elif 'baichuan' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"] # erase bos_token_id
        if target_tok[0] == tok.unk_token_id or hparams.alg_name == 'SERAC':
            target_tok = target_tok[1:]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        textual_ans, textual_target = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    
    ret = {
        f"{locality_key}_output_{lang}": {
            "ans": textual_ans,
            "target": textual_target
        }
    }
    return ret


def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new_en, target_new_zh, ground_truth = (
        record[x] for x in ["target_new_en", "target_new_zh", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts_en = record["rephrase_prompt_en"] if 'rephrase_prompt_en' in record.keys() else None
    rephrase_prompts_zh = record["rephrase_prompt_zh"] if 'rephrase_prompt_zh' in record.keys() else None


    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rewrite_prompts, target_new_en, device=device, lang="en")

    ret['locality_en'] = {}
    ret['locality_zh'] = {}
    ret['portability_en'] = {}
    ret['portability_zh'] = {}
    if rephrase_prompts_en is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rephrase_prompts_en, target_new_en, device=device, test_rephrase=True, lang="en")
        )

    if rephrase_prompts_zh is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, rephrase_prompts_zh, target_new_zh, device=device, test_rephrase=True, lang="zh")
        )

    if 'locality_en' in record.keys() and any(record['locality_en']):
        for locality_key in record['locality_en'].keys():
            ret['locality_en'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality_en'][locality_key]['prompt'],
                                         record['locality_en'][locality_key]['ground_truth'], device=device, lang="en")
            )

    if 'locality_zh' in record.keys() and any(record['locality_zh']):
        for locality_key in record['locality_zh'].keys():
            ret['locality_zh'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality_zh'][locality_key]['prompt'],
                                         record['locality_zh'][locality_key]['ground_truth'], device=device, lang="zh")
            )


    if 'portability_en' in record.keys() and any(record['portability_en']):
        for portability_key in record['portability_en'].keys():
            ret['portability_en'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability_en'][portability_key]['prompt'],
                                            record['portability_en'][portability_key]['ground_truth'], device=device)
            )

    if 'portability_zh' in record.keys() and any(record['portability_zh']):
        for portability_key in record['portability_zh'].keys():
            ret['portability_zh'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability_zh'][portability_key]['prompt'],
                                            record['portability_zh'][portability_key]['ground_truth'], device=device)
            )
    # Form a list of lists of prefixes to test.

    return ret


def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cudAnswer:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        # if locality:
        #     return ans
        
        textual_ans = tok.decode(ans, skip_special_tokens=True)
        textual_target = tok.decode(target, skip_special_tokens=True)

        return textual_ans, textual_target
        # return np.mean(np.equal(ans, target)), textual_ans, textual_target

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target, device, locality=False):
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cudAnswer:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cudAnswer:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]
