from sentence_transformers import SentenceTransformer
import pickle
from torch.utils.data import Dataset
import os
from .ike_hparams import IKEHyperParams


def encode_ike_facts(sentence_model: SentenceTransformer, ds: Dataset, hparams: IKEHyperParams,lang):

    sentences = []
    mt = []
    for i, train_data in enumerate(ds):
        new_fact = train_data['prompt']
        target_new = train_data['target_new']

        sentences.append(new_fact)

        mt.append(f"Question: {train_data['prompt']} Answer: {train_data['target_new']}\nQuestion: {train_data['prompt_mt']} Answer: {train_data['target_new_mt']}\n\n")

    embeddings = sentence_model.encode(sentences)
    base_path = f'{hparams.results_dir}/{hparams.alg_name}/{lang}_embedding'
    os.makedirs(base_path, exist_ok=True)
    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{base_path}/{safe_model_name}_{type(ds).__name__}_{len(ds)}.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings,'mt':mt}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)