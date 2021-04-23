
from torch.nn.modules import loss
import torch.utils.data
from torch.autograd import Variable

from odata import load_data

import rdata

from PIL import Image, ImageDraw
def labelTovertex(labels):
    ver = []
    for label in labels:
        if(label == 12544):
            break;
        vertex = ((label%112)*2,(label/112)*2)
        ver.append(vertex)
    return ver
def label2vertex(labels):
    vertices = []
    for label in labels:
        if (label == 784):
            break
        vertex = ((label % 28) * 8, (label / 28) * 8)
        vertices.append(vertex)
    return vertices
Dataloader = load_data(5, 'val', 60, 1)

Dataloader2 = rdata.load_data(5,'val',60,1)
for step, data in enumerate(Dataloader):
    ta = Variable(data[4].type(torch.LongTensor))
    target = ta.contiguous().view(-1)
    ver2 = label2vertex(target.numpy())
    print(len(ver2))
    img2 = Image.new('RGB', (224, 224), 0)
    ImageDraw.Draw(img2).polygon(ver2, outline="blue", fill=(255, 0, 0))
    img2.save("./save_img/oldta_{}_pred.PNG".format(step),"PNG")

for step, data in enumerate(Dataloader2):
    ta = Variable(data[4].type(torch.LongTensor))
    target = ta.contiguous().view(-1)
    ver2 = labelTovertex(target.numpy())
    print(len(ver2))
    img2 = Image.new('RGB', (224, 224), 0)
    ImageDraw.Draw(img2).polygon(ver2, outline="blue", fill=(255, 0, 0))
    img2.save("./save_img/nowta_{}_pred.PNG".format(step),"PNG")