import config
import os
import json
import os.path
from nn_models.trainer import Trainer, AttentionTrainer
from nn_models.predict import Predictor, AttentionPredictor

# =================attention by mention====================
# --------------------训练---------------
attn_trainer = AttentionTrainer(batch_size=128, n_epoches=20, out_dir=config.attention_mgcn_elstm_model,
                                train_data_fp=os.path.join(config.train_dir, 'train_data_m_e_pair_sort_by_name.json'))
attn_trainer.run()


# attn_predictor = AttentionPredictor(os.path.join(config.attention_mgcn_elstm_model, 'best_model'),
#                                     config.TabEL_mention_abstract_emb_fp)

