import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "aspanformer"))
sys.path.insert(0, ROOT_DIR)

import warnings
import torch
import numpy as np

from evaluator import KeypointModel
from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config


class ASpanFormerModel(KeypointModel):
    CONFIG_PATH = "models/ASpanFormer/aspanformer/configs/aspan/outdoor/aspan_test.py"
    WEIGHTS_PATH = "models/ASpanFormer/aspanformer/weights/outdoor.ckpt"

    def __init__(self):
        config = get_cfg_defaults()
        config.merge_from_file(self.CONFIG_PATH)
        _config = lower_config(config)
        self.matcher = ASpanFormer(config=_config['aspan'])
        state_dict = torch.load(self.WEIGHTS_PATH, map_location='cpu', weights_only=False)['state_dict']
        self.matcher.load_state_dict(state_dict,strict=False)
        self.matcher.cuda()
        self.matcher.eval()

    def find_matching_points(self, img1: np.ndarray, img2: np.ndarray) -> (np.ndarray, np.ndarray):
        data = {
            'image0': img1,
            'image1': img2
        }

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.matcher(data, online_resize=True)
            points1 = data['mkpts0_f'].cpu().numpy()
            points2 = data['mkpts1_f'].cpu().numpy()

        return (points1, points2)
