from easyeditor import MENDHyperParams, SERACHparams
from easyeditor import EditTrainer
from easyeditor import ZsreDataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--editing_method', required=True, type=str)
parser.add_argument("--backbone", type=str, default="chinese_llama7b")
parser.add_argument("--source_lang", type=str, default="en")
args = parser.parse_args()

if args.editing_method == "MEND":
    if args.backbone == "chinese_llama7b":
        training_hparams = MENDHyperParams.from_hparams('hparams/MEND/llama-7b-train')
    elif args.backbone == "chinese_llama2":
        training_hparams = MENDHyperParams.from_hparams('hparams/MEND/llama2-7b-train')
    elif args.backbone == "baichuan7b":
        training_hparams = MENDHyperParams.from_hparams('hparams/MEND/baichuan-7b-train')
    else:
        raise NotImplementedError()
elif args.editing_method == "SERAC":
    if args.backbone == "chinese_llama7b":
        training_hparams = SERACHparams.from_hparams('hparams/SERAC/llama-7b-train')
    elif args.backbone == "chinese_llama2":
        training_hparams =  SERACHparams.from_hparams('hparams/SERAC/llama2-7b-train')
    elif args.backbone == "baichuan7b":
        training_hparams = SERACHparams.from_hparams('hparams/SERAC/baichuan-7b-train')
    else:
        raise NotImplementedError()
else:
    raise NotImplementedError()

if args.source_lang == "en":
    train_ds = ZsreDataset('data/zsre_mend_train_10000.json', config=training_hparams)
    eval_ds = ZsreDataset('data/zsre_mend_eval.json', config=training_hparams)
elif args.source_lang == "zh":
    train_ds = ZsreDataset('data/zsre_mend_train_10000_chinese.json', config=training_hparams)
    eval_ds = ZsreDataset('data/zsre_mend_eval_chinese.json', config=training_hparams)
else:
    raise NotImplementedError()

trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()