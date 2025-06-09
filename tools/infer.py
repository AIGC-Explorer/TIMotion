import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning as L
import scipy.ndimage.filters as filters

from os.path import join as pjoin
from models import *
from collections import OrderedDict
from configs import get_config
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil

import argparse


class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # train model init
        self.model = model

        # others init
        self.normalizer = MotionNormalizer()

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=10)


    def generate_one_sample(self, prompt, name, window_size=210, k=None, gt=None):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt"] = prompt

        # window_size = 210
        motion_output = self.generate_loop(batch, window_size)
        result_path = f"results/{name}.mp4"
        if not os.path.exists("results"):
            os.makedirs("results")
        
        if gt is not None:
            result_gt_path = f"results/{name}_gt.mp4"
            self.plot_t2m([gt[k, :window_size, :66].reshape(-1,22,3), gt[k, :window_size, 262:262+66].reshape(-1,22,3)],
                        result_gt_path,
                        batch["prompt"])
        
        self.plot_t2m([motion_output[0], motion_output[1]],
                      result_path,
                      batch["prompt"])

    def generate_loop(self, batch, window_size):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        sequences = [[], []]

        batch["text"] = [prompt]
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())


        for j in range(2):
            motion_output = motion_output_both[:,j]

            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            sequences[j].append(joints3d)


        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences

def build_models(cfg):
    if cfg.NAME == "TIMotion":
        model = TIMotion(cfg)
    return model


def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--exp-name', default='TIMotion', type=str)
    parser.add_argument('--n-head', default=16, type=int)
    parser.add_argument('--n-layer', default=5, type=int)
    parser.add_argument('--LPA', action='store_true', help='whether use LPA')
    parser.add_argument('--conv-layers', default=1, type=int)
    parser.add_argument('--dilation-rate', default=1, type=int)
    parser.add_argument("--norm", type=str, default='AdaLN', choices = ['AdaLN', 'LN', 'BN', 'GN'])
    parser.add_argument('--latent-dim', default=512, type=int)
    parser.add_argument("--pth", type=str, default=None, help='resume pth')
    
    return parser.parse_args()


if __name__ == '__main__':
    # torch.manual_seed(37)
    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    args = get_args_parser()
    model_cfg.LPA = args.LPA
    model_cfg.conv_layers = args.conv_layers
    model_cfg.dilation_rate = args.dilation_rate
    model_cfg.norm = args.norm
    exp_name = args.exp_name

    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, infer_cfg).to(torch.device("cuda:0"))


    # with open("./prompts.txt") as f:
    with open(exp_name) as f:
        texts = f.readlines()
    texts_list = []
    motion_len_list = []
    for text in texts:
        tmp_text = text.strip("\n")
        # import pdb; pdb.set_trace()
        infos = tmp_text.split('#')
        texts_list.append(infos[0])
        motion_len_list.append(int(infos[-1]))
    # texts = [text.strip("\n") for text in texts]

    motion_gt = None

    for k, text in enumerate(texts_list):
        name = text[:48]
        for i in range(3):
            if i == 0:
                litmodel.generate_one_sample(text, name+"_"+str(i), motion_len_list[k], k, motion_gt)
            else:
                litmodel.generate_one_sample(text, name+"_"+str(i), motion_len_list[k])

