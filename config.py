import os
import torch
from recbole.config import Config as RecBoleConfig

class Config(RecBoleConfig):
    def _init_device(self):
        use_gpu = self.final_config_dict.get('use_gpu', True)
        if use_gpu and ('ddp' in self.final_config_dict and not self.final_config_dict['ddp']):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        # Force CPU if use_gpu is False
        if not use_gpu:
            self.final_config_dict['device'] = torch.device("cpu")
        else:
            self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure single process mode (disable distributed training)
        self.final_config_dict['single_spec'] = True
        self.final_config_dict['ddp'] = False
