import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.INFO)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def _save_checkpoint(self, epoch_num, arch, model, optimizer, scheduler, is_best):
        file_name = "model_best.pth.tar" if is_best else f"checkpoint_{epoch_num}.pth.tar"
        save_checkpoint({
            'epoch': epoch_num,
            'arch': arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best=True, filename=os.path.join(self.writer.log_dir, file_name))

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Disable gpu: {self.args.disable_cuda}.")
        max_top1 = 0
        for epoch_counter in range(1, self.args.epochs + 1):
            loss_epoch = acc1_epoch = acc5_epoch = 0
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_epoch += loss
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                acc1_epoch += top1[0]
                acc5_epoch += top5[0]
                n_iter += 1

            if epoch_counter % 100 == 0:
                self._save_checkpoint(epoch_counter, self.args.arch, self.model, self.optimizer, self.scheduler, False)

            self.writer.add_scalar('loss', loss_epoch, global_step=epoch_counter)
            self.writer.add_scalar('acc/top1', acc1_epoch / len(train_loader), global_step=epoch_counter)
            self.writer.add_scalar('acc/top5', acc5_epoch / len(train_loader), global_step=epoch_counter)
            self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr(), global_step=epoch_counter)
            logging.info(f"Epoch: {epoch_counter}\t"
                         + f"Loss: {loss_epoch}\t"
                         + f"Top1 accuracy: {acc1_epoch / len(train_loader)}")

            if (acc1_epoch / len(train_loader)) > max_top1:
                max_top1 = acc1_epoch / len(train_loader)
                self._save_checkpoint(epoch_counter, self.args.arch, self.model, self.optimizer, self.scheduler, True)

            self.scheduler.step()

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
