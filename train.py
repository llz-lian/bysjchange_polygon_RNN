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

    print('Using gpus: {}'.format(devices))
    torch.cuda.set_device(devices[0])

    Dataloader = load_data(num, 'val', 60, batch_size)
    len_dl = len(Dataloader)
    print(len_dl)

    net = PolygonNet()
    net = nn.DataParallel(net, device_ids=devices)
    #net.load_state_dict(torch.load('./checkpoint/train1_1.pth'))
    


    
    net.cuda()
    print('Loading completed!')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[1, 2],
                                               gamma=0.1)

    dtype = torch.cuda.FloatTensor
    dtype_t = torch.cuda.LongTensor
    sum_loss = 0
    sum_acc = 0
    epoch_r = int(100000 / len_dl)
    for epoch in range(epoch_r):
        for step, data in enumerate(Dataloader):#4->
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
            sum_loss += loss
            loss.backward()

            result_index = torch.argmax(result, 1)
            correct = (target == result_index).type(dtype).sum().item()
            acc = correct * 1.0 / target.shape[0]
            sum_acc += acc
            #        scheduler.step(loss)
            optimizer.step()

            if step % 100 == 0:
                torch.save(net.state_dict(),
                           prefix + '_' + str(step) + '.pth')
                print("loss:{} acc:{}".format(loss,acc))
                # for param_group in optimizer.param_groups:
                #     print(
                #         'epoch{} step{}:{}'.format(epoch, step,
                # param_group['lr']))
        train_iou = test(net, 'train',10)
        val_iou = test(net, 'val',10)
        torch.save(net.state_dict(),
                           prefix + '_' + str(5) + '.pth')
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
"""
loss:9.437235832214355 acc:0.0
loss:9.352361679077148 acc:0.09866848753840901
loss:9.345765113830566 acc:0.09847315148395952
loss:9.342286109924316 acc:0.0996677740863787
loss:9.341378211975098 acc:0.09942815375354702
loss:9.342912673950195 acc:0.09720214742927918
loss:9.342623710632324 acc:0.09703654828159937
loss:9.342436790466309 acc:0.09689360027546826
loss:9.34245777130127 acc:0.0966249085195227
loss:9.34220027923584 acc:0.0966933292510235
loss:9.342293739318848 acc:0.09645526886906171
loss:9.343042373657227 acc:0.09558708384227482
loss:9.342676162719727 acc:0.09585402968790352
loss:9.342863082885742 acc:0.09558297330965543
loss:9.34277629852295 acc:0.09559674124393887
loss:9.342689514160156 acc:0.09562016127179578
loss:9.341930389404297 acc:0.09632449546619573
loss:9.341315269470215 acc:0.09689026738835177
loss:9.340978622436523 acc:0.09718738631794607
loss:9.341194152832031 acc:0.09693627673275425
loss:9.341354370117188 acc:0.09674042289200253
loss:9.341114044189453 acc:0.09694890774508062
loss:9.341238975524902 acc:0.09679769383822433
loss:9.341202735900879 acc:0.09680573663624553
loss:9.340962409973145 acc:0.09702135604417744
loss:9.340842247009277 acc:0.09711977278054344
loss:9.340936660766602 acc:0.09700181627755958
loss:9.341300010681152 acc:0.09661811078910833
loss:9.341530799865723 acc:0.0963633677627459
loss:9.341442108154297 acc:0.0964352363632054
loss:9.341012954711914 acc:0.09684990060784483
loss:9.340876579284668 acc:0.09697094374451129
loss:9.34091854095459 acc:0.09691475724181209
loss:9.340865135192871 acc:0.09695337880893146
loss:9.340744018554688 acc:0.09706323697898402
loss:9.341203689575195 acc:0.09659309162899508
loss:9.341203689575195 acc:0.09657757902498604
loss:9.341222763061523 acc:0.09655358756720264
loss:9.341293334960938 acc:0.0964764263487858
loss:9.341306686401367 acc:0.09645183816705118
loss:9.341431617736816 acc:0.09632290203311546
loss:9.341487884521484 acc:0.09626121467430464
loss:9.3414945602417 acc:0.09624555729752692
loss:9.341535568237305 acc:0.09619655412935592
loss:9.341394424438477 acc:0.09633390530365712
loss:9.34139347076416 acc:0.09632916823081825
loss:9.341341972351074 acc:0.0963733521198578
loss:9.341227531433105 acc:0.09648717440897035
loss:9.341402053833008 acc:0.09630536741627495
loss:9.341270446777344 acc:0.09643176269445757
"""