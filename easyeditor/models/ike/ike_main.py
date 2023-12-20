from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import json
from torch.utils.data import Dataset
from .ike_hparams import IKEHyperParams
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch




def apply_ike_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: IKEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    lang='en',
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    assert train_ds is not None
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)

    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{hparams.results_dir}/{hparams.alg_name}/{lang}_embedding/'
              f'{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_embeddings = stored_data['embeddings']
        stored_mt = stored_data['mt']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    stored_embeddings = util.normalize_embeddings(stored_embeddings)
    if request['search_prompt'] != "":

        new_fact = request['search_prompt']

        query_sentence = new_fact

        query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
            query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
        assert len(hits) == 1
        hit = hits[0]

        icl_examples = [stored_mt[hit[k]["corpus_id"]] for k in range(len(hit))]
    else:
        icl_examples = ['']


    return icl_examples
