import json
from PIL import Image, ImageDraw
import numpy as np
from torch.nn import parameter
import torch.utils.data
from torch import  nn
from torch.autograd import Variable, backward
import time
from nets.omodel import PolygonNet
from utils.utils import img2tensor
from utils.utils import getbboxfromkps
from utils.utils import  getbboxfromkps
import glob
import cv2
from datas.data import load_data
dtype = torch.cuda.FloatTensor
drawing = False
x1,y1 = -1,-1
selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                        'rider', 'bus', 'train']
def fpstest(net):
    file = 'img/val/lindau/lindau_000000_000019_leftImg8bit.png'
    json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
    json_object = json.load(open(json_file))
    h = json_object['imgHeight']
    w = json_object['imgWidth']
    objects = json_object['objects']
    img = Image.open(file).convert('RGB')
    I = np.array(img)
    res = []
    for obj in objects:
        if obj['label'] in selected_classes:
            min_row, min_col, max_row, max_col = getbboxfromkps(
            obj['polygon'], h, w)
            object_h = max_row - min_row
            object_w = max_col - min_col
            scale_h = 224.0 / object_h
            scale_w = 224.0 / object_w
            I_obj = I[min_row:max_row, min_col:max_col, :]
            I_obj_img = Image.fromarray(I_obj)
            I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
            I_obj_new = np.array(I_obj_img)
            xx = img2tensor(I_obj_new)
            xx = xx.unsqueeze(0).type(dtype)
            
            xx = Variable(xx)
            torch.cuda.synchronize()
            start = time.time()
            re = net.test(xx, 60)
            torch.cuda.synchronize()
            end = time.time()
            res.append(end-start)
    time_sum = 0
    for i in res:
        time_sum += i
    print("FPS: %f"%(1.0/(time_sum/len(res))))
def getNet():
    torch.cuda.set_device(0)
    net = PolygonNet(load_vgg=False)
    net = nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load('./VGG40.pth'))
    net.cuda()
    return net

def predictFromJson():
    net = getNet()
    print('Loading completed!')

    json_file = json.load(open('./label/2.json'))
    objects = json_file['shapes']
    img = Image.open('img/6C471B3279340E0CDEF4C2E3FA29C2F2.png').convert('RGB')
    I = np.array(img)
    set_class = []
    for obj in objects:
        if obj['label'] in selected_classes:
            min_row, min_col, max_row, max_col = getbboxfromkps(
                    obj['points'], 3024,4032)

            object_h = max_row - min_row
            object_w = max_col - min_col
            scale_h = 224.0 / object_h
            scale_w = 224.0 / object_w
            I_obj = I[int(min_row):int(max_row),int(min_col):int(max_col), :]

            I_obj_img = Image.fromarray(I_obj)
            I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
            I_obj_new = np.array(I_obj_img)

            xx = img2tensor(I_obj_new)
            xx = xx.unsqueeze(0).type(dtype)
            xx = Variable(xx)

            re,classes,ret = net.module.test(xx, 3)
            print(ret)
            time.sleep(10000)
            indx = classes.argmax()
            classes = selected_classes[indx] 
            set_class.append(classes)   
            labels_p = re.cpu().numpy()[0]
            vertices1 = []  
            for label in labels_p:
                if (label == 784):
                    break
                vertex = (
                ((label % 28) * 8.0 + 4) / scale_w + min_col, 
                ((int(label / 28)) * 8.0 + 4) / scale_h + min_row
                )
                vertices1.append(vertex)

            color = [np.random.randint(0, 255) for _ in range(3)]
            color += [100]
            color = tuple(color)
            

def old_predict(imga):
    image = cv2.imread(imga)
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        global x1,y1,x2,y2,drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1,y1 = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            pass
        elif event==cv2.EVENT_LBUTTONUP:
            drawing = False
            x2,y2 = x,y
    net = getNet()
    set_class = []
    global x1,y1,x2,y2
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image',on_EVENT_LBUTTONDOWN)
    json_file = {}
    json_file['w'] = image.shape[0]
    json_file['h'] = image.shape[1]
    objs = []
    num  = 3
    img = Image.open(imga).convert('RGB')
    while(num>0):
        num = num -1;
        obj = {}
        while(1):
            cv2.imshow('image',image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        print(x1,y1)
        print(x2,y2)
        obj_points = [(x1,y1),(x2,y2)]
        min_row, min_col, max_row, max_col = getbboxfromkps(
                        obj_points, image.shape[0],image.shape[1])
        object_h = max_row - min_row
        object_w = max_col - min_col
        scale_h = 224.0 / object_h
        scale_w = 224.0 / object_w
        I = np.array(img)
        I_obj = I[int(min_row):int(max_row),int(min_col):int(max_col), :]
        I_obj_img = Image.fromarray(I_obj)
        I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
        I_obj_new = np.array(I_obj_img)
        xx = img2tensor(I_obj_new)
        xx = xx.unsqueeze(0).type(dtype)
        xx = Variable(xx)
        re,classes = net.module.test(xx, 60)
        indx = classes.argmax()
        classes = selected_classes[indx] 
        obj['classes'] = classes
        set_class.append(classes)   
        labels_p = re.cpu().numpy()[0]
        vertices1 = []  
        for label in labels_p:
            if (label == 784):
                break
            vertex = (
            ((label % 28) * 8.0 + 4) / scale_w + min_col, 
            ((int(label / 28)) * 8.0 + 4) / scale_h + min_row
            )
            vertices1.append(vertex)
        obj['points'] = vertices1
        objs.append(obj)
        color = [np.random.randint(0, 255) for _ in range(3)]
        color += [100]
        color = tuple(color)
        drw = ImageDraw.Draw(img, 'RGBA')
        drw.polygon(vertices1, color)
    file_name = './myresult/test.json'
    json_file['objects'] = objs
    with open(file_name,'w') as f:
        json.dump(json_file,f)
    img.save('./myresult/image_test.png','PNG')

def getnum(dataset):
    dic = {}
    files = glob.glob('./label/*/*/*.json'.format(dataset))
    nums = [0 for i in range(len(selected_classes))]
    avgnum = [0 for i in range(len(selected_classes))]
    for file in files:
        json_file = json.load(open(file))
        objs = json_file["objects"]
        for obj in objs:
            if obj['label'] in selected_classes:
                index = selected_classes.index(obj['label'])
                nums[index] = nums[index] +1
                avgnum[index] = avgnum[index] + len(obj['polygon'])
    indx = 0
    num = np.array(nums)
    avgnum = np.array(avgnum)
    avgnum = avgnum/num
    print(avgnum)
    print(num)
    for i in selected_classes:
        dic[i] = (nums[indx],avgnum[indx])
        indx = indx+1
    return dic
def sumPolygonPoints(dataset):
    files = glob.glob('./label/{}/*/*.json'.format(dataset))
    nums = np.zeros((len(selected_classes),8),dtype=np.uint8)#nums[i] :   x<40  x<80 x<120 x<160 x<200 x<240 others
    for file in files:
        json_file = json.load(open(file))
        objs = json_file["objects"]
        for obj in objs:
            if obj['label'] in selected_classes:
                index = selected_classes.index(obj['label'])
                points = len(obj['polygon'])
                q = int(points/40)
                if points/40>7:
                    q = 7
                nums[int(index),q] =  nums[int(index),q] +1
    print(nums)
    for i in range(len(selected_classes)):
        print('nums of polygon:')
        print(selected_classes[i])
        for j in range(8):
            print('   <{} '.format((j+1)*40),end=' ')
            print(nums[i][j],end=' ')
        print()







def predict(image_name):
    original_image = cv2.imread(image_name)
    back_ground = original_image
    net = getNet()
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        global x1,y1,x2,y2,drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1,y1 = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            pass
        elif event==cv2.EVENT_LBUTTONUP:
            drawing = False
            x2,y2 = x,y
    global x1,y1,x2,y2
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image',on_EVENT_LBUTTONDOWN)
    num = 6
    
    while(num>0):
        num = num -1;
        while(1):
            cv2.imshow('image',original_image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        print(x1,y1)
        print(x2,y2)
        obj_points = [(x1,y1),(x2,y2)]
        min_row, min_col, max_row, max_col = getbboxfromkps(
                            obj_points, original_image.shape[0],original_image.shape[1])
        object_h = max_row - min_row
        object_w = max_col - min_col
        scale_h = 224.0 / object_h
        scale_w = 224.0 / object_w
        scale_image = original_image[int(min_row):int(max_row),int(min_col):int(max_col), :]
        scale_image = cv2.resize(scale_image,(224,224))

        b,g,r = cv2.split(scale_image)
        rgb_pic = cv2.merge([r,g,b])
        xx = img2tensor(rgb_pic)
        xx = Variable(xx)
        xx = xx.unsqueeze(0).type(dtype)#add batch
        net.module.eval()
        re = net.module.test(xx, 60)
        labels_p = re.cpu().numpy()[0]
        vertices1 = []  
        for label in labels_p:
            if (label == 784):
                break
            vertex = (
            ((label % 28) * 8.0 + 4) / scale_w + min_col, 
            ((int(label / 28)) * 8.0 + 4) / scale_h + min_row
            )
            vertices1.append(vertex)
        color = [np.random.randint(0, 255) for _ in range(3)]
        color += [100]
        #color = tuple(color)

        vertices1 = np.array(vertices1,dtype=np.int)
        
        back_ground = cv2.fillConvexPoly(back_ground,
                                    vertices1,
                                    (100,100,220)
                                    )
        original_image = cv2.addWeighted(back_ground,0.2,original_image,0.8,0)
predict("img/6C471B3279340E0CDEF4C2E3FA29C2F2.png")


"""
classes nums:
#train:{'person': 17994, 'car': 27155, 'truck': 489, 'bicycle': 3729, 'motorcycle': 739, 'rider': 1807, 'bus': 385, 'train': 171}
#val:{'person': 3419, 'car': 4667, 'truck': 93, 'bicycle': 1175, 'motorcycle': 149, 'rider': 556, 'bus': 98, 'train': 23}
{'person': (21413, 31.38117031709709), 'car': (31822, 24.51486393061404), 'truck': (582, 28.644329896907216), 'bicycle': (4904, 34.241231647634585), 'motorcycle': (888, 38.4695945945946), 'rider': (2363, 37.63817181548878), 'bus': (483, 31.05175983436853), 'train': (194, 33.118556701030926)}
nums of polygon:
person
   <40  199    <80  10    <120  165    <160  157    <200  39    <240  11    <280  5    others  0 avg
car
   <40  122    <80  141    <120  251    <160  16    <200  0    <240  0    <280  1    others  0
truck
   <40  118    <80  94    <120  14    <160  2    <200  2    <240  2    <280  1    others  0
bicycle
   <40  20    <80  170    <120  164    <160  35    <200  8    <240  3    <280  1    others  0
motorcycle
   <40  210    <80  212    <120  52    <160  7    <200  1    <240  0    <280  1    others  0 
rider
   <40  95    <80  50    <120  112    <160  10    <200  3    <240  1    <280  0    others  0
bus
   <40  21    <80  90    <120  16    <160  1    <200  1    <240  0    <280  0    others  0
train
   <40  124    <80  37    <120  7    <160  1    <200  1    <240  0    <280  0    others  1
"""