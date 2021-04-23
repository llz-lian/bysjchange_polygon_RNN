import argparse

import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.autograd import Variable
from thop import profile
from config_tools import get_config
from data import load_data
from model import PolygonNet
from test import test
import time

def train(config, pretrained=None):
    devices = config['gpu_id']
    batch_size = config['batch_size']
    lr = config['lr']
    log_dir = config['log_dir']
    prefix = config['prefix']
    num = config['num']



    Dataloader = load_data(num, 'val', 60, batch_size)
    len_dl = len(Dataloader)
    print(len_dl)

    net = PolygonNet()


    #if pretrained:
    #    net.load_state_dict(torch.load(pretrained))

    print('Loading completed!')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[4000, 100000],
                                               gamma=0.1)

    dtype = torch.FloatTensor
    dtype_t = torch.LongTensor

    epoch_r = int(20000 / len_dl)
    for epoch in range(epoch_r):
        for step, data in enumerate(Dataloader):
            scheduler.step()
            x = Variable(data[0].type(dtype))
            x1 = Variable(data[1].type(dtype))
            x2 = Variable(data[2].type(dtype))
            x3 = Variable(data[3].type(dtype))
            ta = Variable(data[4].type(dtype_t))
            optimizer.zero_grad()
            r = net(x, x1, x2, x3)
            result = r.contiguous().view(-1, 112 * 112+3)

            target = ta.contiguous().view(-1)
            #torch.Size([58])
            #torch.Size([58, 12544])
            loss = loss_function(result, target)

            loss.backward()

            result_index = torch.argmax(result, 1)
            correct = (target == result_index).type(dtype).sum().item()
            acc = correct * 1.0 / target.shape[0]

            #        scheduler.step(loss)
            optimizer.step()

            if step % 100 == 0:
                torch.save(net.state_dict(),
                           prefix + '_' + str(num) + '.pth')
                print("loss:{} acc:{}".format(loss,acc))
                # for param_group in optimizer.param_groups:
                #     print(
                #         'epoch{} step{}:{}'.format(epoch, step,
                # param_group['lr']))
        train_iou = test(net, 'train',10)
        val_iou = test(net, 'val',10)
        print('iou score on training set:{}'.format(train_iou))
        print('iou score on test set:{}'.format(val_iou))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-l', dest='log_dir', type=str,
                        help='Location of Logs')
    parser.add_argument('--gpu_id', '-g', type=int, nargs='+', help='GPU Id')
    parser.add_argument('--batch_size', '-b', type=int, help='Batch Size')
    parser.add_argument('--num', '-n', type=int, help='Number of Instances')
    parser.add_argument('--lr', type=float, help='Learning Rate')
    parser.add_argument('--prefix', type=str, help='Model Prefix')
    parser.add_argument('--pretrained', '-p', type=str, help='Pretrained '
                                                             'Model Location')
    parser.add_argument('--config', dest='config_file', help='Config File')
    args = parser.parse_args()
    config_from_args = args.__dict__
    config_file = config_from_args.pop('config_file')
    config = get_config('train', config_from_args, config_file)
    train(config, config['pretrained'])
