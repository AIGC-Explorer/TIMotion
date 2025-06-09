import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *

import argparse
import random

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, start_saving_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_saving_epoch = start_saving_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_saving_epoch-1:
            super().on_train_epoch_end(trainer, pl_module)


class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        # TODO
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=2000, verbose=True)
        # scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        name, text, motion1, motion2, motion_lens = batch_data
        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])



    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()


    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


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
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--drop-out', default=0.1, type=float)
    parser.add_argument('--LPA', action='store_true', help='whether use LPA')
    parser.add_argument('--conv-layers', default=1, type=int)
    parser.add_argument('--dilation-rate', default=1, type=int)
    parser.add_argument("--norm", type=str, default='AdaLN', choices = ['AdaLN', 'LN', 'BN', 'GN'])
    parser.add_argument('--latent-dim', default=512, type=int)
    parser.add_argument("--resume", type=str, default=None, help="eps for optimal transport")
    parser.add_argument('--epoch', default=1900, type=int)
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    
    return parser.parse_args()

if __name__ == '__main__':
    print(os.getcwd())
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml").interhuman

    args = get_args_parser()
    
    fixseed(args.seed)
    
    train_cfg.GENERAL.EXP_NAME = args.exp_name
    model_cfg.NUM_HEADS = args.n_head
    model_cfg.NUM_LAYERS = args.n_layer
    model_cfg.DROPOUT = args.drop_out

    model_cfg.LPA = args.LPA
    model_cfg.conv_layers = args.conv_layers
    model_cfg.dilation_rate = args.dilation_rate
    model_cfg.norm = args.norm
    model_cfg.LATENT_DIM = args.latent_dim
        
    train_cfg.LR = args.lr
    train_cfg.BATCH_SIZE = args.batch_size
    if args.resume is not None:
        train_cfg.TRAIN.RESUME = args.resume

    datamodule = DataModule(data_cfg, train_cfg.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_models(model_cfg)


    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg)

    checkpoint_callback = CustomModelCheckpoint(start_saving_epoch=1200,
                                                       dirpath=litmodel.model_dir,
                                                       every_n_epochs=100,
                                                    #    every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k=-1,  # 保存所有检查点，不覆盖之前的结果
                                                       )

    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices="auto", accelerator='gpu',
        max_epochs=args.epoch,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],

    )

    trainer.fit(model=litmodel, datamodule=datamodule)
