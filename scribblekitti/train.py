import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import ConfusionMatrix

from network.cylinder3d import Cylinder3D
from dataloader.semantickitti import SemanticKITTI
from utils.lovasz import lovasz_softmax
from utils.consistency_loss import PartialConsistencyLoss
from utils.evaluation import compute_iou

# 19 classes
class_name = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
              'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
              'traffic-sign']
class LightningTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.student = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.teacher = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.initialize_teacher()

        self.loss_ls = lovasz_softmax
        self.loss_cl = PartialConsistencyLoss(H=nn.CrossEntropyLoss, ignore_index=0)
        print('initialization', self.global_rank)
        self.teacher_cm = ConfusionMatrix(self.nclasses)
        self.student_cm = ConfusionMatrix(self.nclasses)
        self.best_miou = 0
        self.best_iou = np.zeros((self.nclasses-1,))
        self.gpu_num = len(config['trainer']['devices'])
        self.save_hyperparameters('config')

    def forward(self, model, fea, pos, bs):
        # fea: list
        # pos: list
        # output_voxel = model([fea.squeeze(0)], [pos.squeeze(0)], bs)
        output_voxel = model(fea, pos, bs)
        # print('shape output_voxel', output_voxel.shape)   # [2,20,480,360,32]
        B = output_voxel.shape[0]
        # print('pos[0]', pos[0].shape, pos[0][0])     # [2,124xxx,3], len(pos)=2
        outputs = [output_voxel[i, :, pos[i][:,0], pos[i][:,1], pos[i][:,2]] for i in np.arange(B)]
        return outputs

    def training_step(self, batch, batch_idx):
        # print('enter training step', self.global_rank, 'self.device', self.device) 0,1,2,3
        if self.global_rank == 0:
            self.update_teacher()
        student_rpzb, student_feab, student_labelb = batch['student']
        teacher_rpzb, teacher_feab, _ = batch['teacher']

        student_fea_ten = [i for i in student_feab]
        # print('get device of student_fea_ten if not set', student_fea_ten[0].get_device())
        teacher_fea_ten = [i for i in teacher_feab]
        student_rpz_ten = [i for i in student_rpzb]
        teacher_rpz_ten = [i for i in teacher_rpzb]

        student_label_ten = [i for i in student_labelb]
        batch_size = len(student_fea_ten)

        student_output_batch = self(self.student, student_fea_ten, student_rpz_ten, batch_size)
        # student_output_batch[0].shape = [20, 119616]
        teacher_output_batch = self(self.teacher, teacher_fea_ten, teacher_rpz_ten, batch_size)
        loss = torch.tensor([0.0], device=self.device)

        for i in np.arange(batch_size):
            student_label = student_label_ten[i].unsqueeze(0)
            student_output, teacher_output = student_output_batch[i].unsqueeze(0), teacher_output_batch[i].unsqueeze(0)
            loss += self.loss_cl(student_output, teacher_output, student_label) + \
               self.loss_ls(student_output.softmax(1), student_label, ignore=0)
        train_loss_gather = self.all_gather(loss).mean()
        # print('loss', loss, 'train_loss_gather', train_loss_gather, 'self.all_gather(loss)', self.all_gather(loss)一个list of len 4)
        if self.global_rank == 0:
            self.log('train_loss', train_loss_gather, on_epoch=True, prog_bar=True, rank_zero_only=True,
                     batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        student_rpzb, student_feab, student_labelb = batch['student']
        teacher_rpzb, teacher_feab, teacher_labelb = batch['teacher']

        student_fea_ten = [i for i in student_feab]
        teacher_fea_ten = [i for i in teacher_feab]
        student_rpz_ten = [i for i in student_rpzb]
        teacher_rpz_ten = [i for i in teacher_rpzb]

        student_label_ten = [i for i in student_labelb]
        teacher_label_ten = [i for i in teacher_labelb]
        batch_size = len(student_fea_ten)

        student_output_batch = self(self.student, student_fea_ten, student_rpz_ten, batch_size)
        teacher_output_batch = self(self.teacher, teacher_fea_ten, teacher_rpz_ten, batch_size)

        loss = torch.tensor([0.0], device=self.device)
        B = len(student_label_ten)
        for i in np.arange(B):
            teacher_label = teacher_label_ten[i].unsqueeze(0)
            student_label = student_label_ten[i].unsqueeze(0)
            student_output, teacher_output = student_output_batch[i].unsqueeze(0), teacher_output_batch[i].unsqueeze(0)
            mask = (teacher_label != 0).squeeze()     # student_label = teacher_label
            # print('update_cm', self.global_rank) 每个进程会经过4次此处 when BS=4
            # check below confusion matrix update with ddp, correct?
            self.student_cm.update(student_output.argmax(1)[:,mask], student_label[:,mask])
            self.teacher_cm.update(teacher_output.argmax(1)[:,mask], teacher_label[:,mask])
            loss += self.loss_cl(student_output, teacher_output, student_label) + \
                   self.loss_ls(student_output.softmax(1), student_label, ignore=0)

        validation_loss_gather = self.all_gather(loss).mean()
        if self.global_rank == 0:
            self.log('val_loss', validation_loss_gather, on_epoch=True, prog_bar=True,
                     batch_size=B)

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            student_iou, student_miou = compute_iou(self.student_cm.compute(), ignore_zero=True)
            self.student_cm.reset()

            teacher_iou, teacher_miou = compute_iou(self.teacher_cm.compute(), ignore_zero=True)
            self.teacher_cm.reset()

            if teacher_miou > self.best_miou:
                self.best_miou = teacher_miou
                self.best_iou = np.nan_to_num(teacher_iou) * 100
            self.log('val_best_miou', self.best_miou, on_epoch=True, prog_bar=True, logger=False)
            for classs, class_iou in zip(self.unique_name, teacher_iou):
                self.log('val_teacher_iou_{}'.format(classs), class_iou * 100)

            file = open('/home/jzhang2297/weaksup/scribblekitti/logs/val_iou.txt', "a")
            for classs, class_iou in zip(class_name, teacher_iou):
                print('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('teacher iou, training step:{0}'.format(self.global_step))
                file.write('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('\n')
            file.write('meanIOU={0}'.format(teacher_miou))
            file.write('\n')
            for classs, class_iou in zip(class_name, student_iou):
                print('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('student iou, training step:{0}'.format(self.global_step))
                file.write('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('\n')
            file.write('meanIOU={0}'.format(student_miou))
            file.write('\n')
            file.close()
        else:
            teacher_miou = torch.tensor(0).to(self.device)
            student_miou = torch.tensor(0).to(self.device)

        if self.gpu_num > 1:
            print('self.gpu_num', self.gpu_num)
            teacher_miou = sum(self.all_gather(teacher_miou))
            student_miou = sum(self.all_gather(student_miou))

        self.log('val_teacher_miou', teacher_miou, on_epoch=True, prog_bar=True)
        self.log('val_student_miou', student_miou, on_epoch=True, prog_bar=True, logger=False)

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), **self.config['optimizer'])
        return [optimizer]

    def setup(self, stage):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['dataset'])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.config['train_dataloader'],
                          collate_fn=self.collate_fn_BEV, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.config['val_dataloader'],
                          collate_fn=self.collate_fn_BEV, pin_memory=True)

    def initialize_teacher(self) -> None:
        self.alpha = 0.99 # TODO: Move to config
        for p in self.teacher.parameters(): p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(alpha).add_(1 - alpha, sp.data)

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = len(dataset_config['labels'])
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)
        for i in range(self.nclasses):
            self.color_map[i,:] = torch.tensor(dataset_config['color_map'][i][::-1], dtype=torch.float32)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'])
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}-{val_teacher_miou:.2f}',
                                                  monitor='val_teacher_miou', mode='max', save_top_k=3)
        return [checkpoint]

    def collate_fn_BEV(self, data):
        stu_rpz2stack = [d['student'][0] for d in data]
        stu_fea2stack = [d['student'][1] for d in data]
        stu_label2stack = [d['student'][2] for d in data]

        tea_rpz2stack = [d['teacher'][0] for d in data]
        tea_fea2stack = [d['teacher'][1] for d in data]
        tea_label2stack = [d['teacher'][2] for d in data]
        #return rpz2stack, fea2stack, label2stack
        return {
                'student': (stu_rpz2stack, stu_fea2stack, stu_label2stack),
                'teacher': (tea_rpz2stack, tea_fea2stack, tea_label2stack)
            }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    wandb_logger = WandbLogger(config=config,
                               save_dir=config['trainer']['default_root_dir'],
                               **config['logger'])
    model = LightningTrainer(config)
    Trainer(logger=wandb_logger,
            callbacks=model.get_model_callback(),
            **config['trainer']).fit(model)