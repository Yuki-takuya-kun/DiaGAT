import yaml
import argparse
from attrdict import AttrDict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from src.common import WordPair, DataProcessor, set_seed
from src.preprocess import DataPreprocessor
from src.dataloader import DiaDataloader
from src.model import DiaGAT



if __name__ == '__main__':
    with open('src/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = AttrDict(cfg)
    if cfg.lang == 'en':
        for k, v in cfg['bert-en'].items():
            setattr(cfg, k, v)
    elif cfg.lang == 'zh':
        for k, v in cfg['bert-zh'].items():
            setattr(cfg, k, v)
            
    cfg['schedulers']['total_epochs'] = cfg['epoch_size']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=bool, help='preprocess dataset or not')
    args = parser.parse_args()
    
    if args.dataset:
        preprocessor = DataPreprocessor(cfg)
        preprocessor.process()
    
    set_seed(cfg.random_seed)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    word_pair = WordPair(cfg.lang)
    dataprocessor = DataProcessor(cfg)
    dia_model = DiaGAT(plm_name=cfg.bert_path, pos_nums=word_pair.pos_num,
                       dep_nums=word_pair.dep_num, token_label_num=word_pair.entity_num,
                       edge_classes=word_pair.relation_num,
                       dataprocessor=dataprocessor, schedulers_cfg=cfg['schedulers'],
                       huggingface_dir=cfg.huggingface_dir,
                       aggregate_method=cfg.aggregate_method,
                       node_loss_weight=cfg['loss_weight']['ent'],
                       edge_loss_weight=cfg['loss_weight']['rel'],
                       polar_loss_weight=cfg['loss_weight']['pol'],
                       gat_layer_num = cfg['ModelCfg']['gat_layer_num'],
                       node_gat_layer_num= cfg['ModelCfg']['node_gat_layer_num'],
                       edge_gat_layer_num=cfg['ModelCfg']['edge_gat_layer_num'],
                       sen_gat_layer_num=cfg['ModelCfg']['sen_gat_layer_num'],
                       plm_abla=cfg['plm_abla'],
                       dropout=cfg['dropout'],
                       num_heads=cfg['num_heads'])
    dataset = DiaDataloader(cfg)
    trainer_cfg = cfg['Trainer']
    trainer = pl.Trainer(accelerator=trainer_cfg['accelerator'], max_epochs=cfg['epoch_size'],
                         devices=trainer_cfg['devices'],
                         accumulate_grad_batches=trainer_cfg['accumulate_grad_batches'],
                         enable_checkpointing=False, callbacks=[lr_monitor])
    trainer.fit(dia_model, datamodule=dataset)
    trainer.test(dia_model, datamodule=dataset)
    
        
    