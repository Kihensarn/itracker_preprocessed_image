from loguru import logger
from visdom import Visdom
from typing import Tuple
from pathlib import Path


class my_logger:
    def __init__(self, args):
        logger.add(Path(args.log_dir).joinpath(args.log_name, 'trian.log'), format='{time:YYYY-MM-DD HH:mm:ss} | {level} - {message}', filter=lambda x: 'train' in x['message'])
        logger.add(Path(args.log_dir).joinpath(args.log_name, 'test.log'), format='{time:YYYY-MM-DD HH:mm:ss} | {level} - {message}', filter=lambda x: 'test' in x['message'])
        logger.add(Path(args.log_dir).joinpath(args.log_name, 'val.log'), format='{time:YYYY-MM-DD HH:mm:ss} | {level} - {message}', filter=lambda x: 'validate' in x['message'])
  
    def train_log_n_step(self, epoch, step, total_step, loss, ang_error):
        logger.info('(train): [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Error {ang_error.val:.4f} ({ang_error.avg:.4f})', epoch, step, total_step,
                    loss=loss, ang_error=ang_error)
      
    def train_log_per_epoch(self, epoch, loss, ang_error, lr):
        logger.info('\n-------------------------------------------------------------------\n'
                '(train):\t\t\t({0})\nLoss\t\t\t({loss.avg:.4f})\n'
                'Error\t\t\t({ang_error.avg:.4f})\nlr\t\t\t({lr_show})\n'
                '-------------------------------------------------------------------',
                epoch, loss=loss, ang_error=ang_error, lr_show=lr)

    def train_log_lr(self, epoch, lr):
        logger.info('\n-------------------------------------------------------------------\n'
                '(train):\t\t\t({0})\nlr\t\t\t({lr_show})\n'
                '-------------------------------------------------------------------',
                epoch, lr_show=lr)


    def val_log_n_step(self, epoch, step, total_step, loss, ang_error):
        logger.info('(validate): [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Error {ang_error.val:.4f} ({ang_error.avg:.4f})', epoch, step, total_step,
                    loss=loss, ang_error=ang_error)

    def val_log_per_epoch(self, epoch, loss, ang_error):
        logger.info('\n-------------------------------------------------------------------\n'
                '(validate):\t\t\t({0})\nLoss\t\t\t({loss.avg:.4f})\n'
                'Error\t\t\t({ang_error.avg:.4f})\n'
                '-------------------------------------------------------------------',
                epoch, loss=loss, ang_error=ang_error)

    def test_log(self, batch, total_batch):
        logger.info('(test) [{}/{}] success to verify this batch', batch, total_batch)
    

class my_visdom:
    def __init__(self):
        self.vis = Visdom()

    def train_vis(self, loss_point, error_point):
        self.vis.line(X=[loss_point[0]],Y=[loss_point[1]],
                 win='loss_train',opts={'title':'entire_train_loss'},update='append')
        self.vis.line(X=[error_point[0]], Y=[error_point[1]],
                 win='error_train', opts={'title': 'entire_train_error'}, update='append')

    def validate_vis(self, loss_point, error_point):
        self.vis.line(X=[loss_point[0]],Y=[loss_point[1]], win='loss_validate',
                 opts={'title': 'entire_validate_loss'}, update='append')
        self.vis.line(X=[error_point[0]], Y=[error_point[1]], win='error_validate',
                 opts={'title': 'entire_validate_error'}, update='append')


