import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import string
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException 
from tencentcloud.ocr.v20181119 import ocr_client, models
import base64
import json
import requests


characters = string.digits + string.ascii_uppercase
alphabet=characters
width, height, n_len, n_classes = 112, 45, 4, len(characters)
n_input_length = 12
cuda=False

def delblank(s):
    res=""
    for i in s:
        if(i!=" "):
            res+=i
    return res

def getBase64(filename):
    f = open(filename, 'rb')
    img = base64.b64encode(f.read())
    res=""
    for i in img:
        res+=chr(i)
    return res

def getTencentLetters(filename,user_id,user_secret):
    try:
        cred = credential.Credential(user_id, user_secret) 
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = ocr_client.OcrClient(cred, "ap-shanghai", clientProfile) 

        req = models.GeneralAccurateOCRRequest()
        base64=getBase64(filename)
        params = '{\"ImageBase64\":\"'+base64+'\"}'
        req.from_json_string(params)

        resp = client.GeneralAccurateOCR(req)
        if(len(json.loads(s=resp.to_json_string())["TextDetections"])>0):
              return json.loads(s=resp.to_json_string())["TextDetections"][0]["DetectedText"]
        else:
              return ""
    except:
        return ""

def tencentPreds(path,user_id,user_secret):
    preds=[]
    for i in os.listdir(path)[:]:
        pred=delblank(getTencentLetters(path+i,user_id,user_secret))
        preds.append(pred)
    return preds

def getBaiduLetters(filename,token):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    # 二进制方式打开图片文件
    f = open(filename, 'rb')
    img = base64.b64encode(f.read())

    params = {"image":img}
    access_token =token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        dict=response.json()
        if(len(dict['words_result'])>0):
            return dict['words_result'][0]['words']
        else:
            return ""

def baiduPreds(path,token):
    preds=[]
    for i in os.listdir(path)[:]:
        pred=delblank(getBaiduLetters(path+i,token))
        preds.append(pred)
    return preds

def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


def calc_acc(target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    preds=[]
    for true, pred in zip(target, output_argmax):
        preds.append(decode(pred))
    a = np.array([decode_target(true) == decode(pred)[:4] for true, pred in zip(target, output_argmax)])
    return a.mean(),preds

def img_loader(img_path):
    img = Image.open(img_path)
    return img

def make_dataset(data_path, alphabet, num_class, num_char):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        target_str = img_name.split('.')[0]
        #assert len(target_str) == num_char
        target=[alphabet.find(x) for x in target_str]
        samples.append((img_path, target))
    return samples

class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=37, num_char=1, 
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = characters
        self.samples = make_dataset(self.data_path, self.alphabet, 
                                    self.num_class, self.num_char)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = to_tensor(img_loader(img_path))
        input_length = torch.full(size=(1, ), fill_value=3, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=1, dtype=torch.long)
        target=torch.tensor(target, dtype=torch.long)
        return img[:3], target,input_length,target_length

class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(1, 45, 112)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 1, (2, 1)]
        modules = OrderedDict()
        
        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)
        
        last_channel = 1
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)
        
        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)
    
    def infer_features(self):
        x = torch.zeros((1,)+self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def valid(model, optimizer, epoch, dataloader):
    model.eval()
    preds=[]
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)            
            loss = loss.item()
            acc,pred = calc_acc(target, output)
            for i in pred:
                preds.append(i)
            loss_sum += loss
            acc_sum += acc
            
            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
    return preds

def myModelPreds(path,modelname):
    transforms = Compose([ToTensor()])
    test_data = CaptchaData(path, transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=74, 
                                    num_workers=12)

    model = torch.load(modelname,map_location=lambda storage, loc: storage)
    model=model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
    preds=valid(model, optimizer, 1, test_data_loader)
    return preds

def isValue(s):
    if(len(s)!=4):
        return False
    for i in s:
        if(i not in alphabet):
            return False
    return True

def getMyPred(p):
    for i in p:
        if(isValue(i)):
            return i
    return ""

path="./t/"
model="./model/capt_5.pth"
baidutoken=''
tencentuser=""
tencentsecret=""



tencentpreds=tencentPreds(path,tencentuser,tencentsecret)
baidupreds=baiduPreds(path,baidutoken)
mypreds=myModelPreds(path,model)
preds=[]
for i in range(len(tencentpreds)):
    preds.append([tencentpreds[i],mypreds[i],baidupreds[i]])
finalpreds=[]
for i in preds:
    finalpreds.append(getMyPred(i))
print(finalpreds)