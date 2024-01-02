import os
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
from networks.FADNet import FADNet
from struct import unpack
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import warnings 
warnings.filterwarnings('ignore')

class sceceflow_detect:
    normalize={'mean': [0.485, 0.456, 0.406],'std': [0.229, 0.224, 0.225]}
    transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ]
        )
    
    def __init__(self,root='sceneflow/FlyingThings3D_release/frames_cleanpass/TEST/A/0000/',pt='models/fadnet.pth'):
        lefts=os.path.join(root,'left')
        rights=os.path.join(root,'right')
        disps=lefts.replace('frames_cleanpass','disparity')
        self.lefts_pt=[os.path.join(lefts,i) for i in os.listdir(lefts)]
        self.rights_pt=[os.path.join(rights,i) for i in os.listdir(rights)]
        self.disp_pt=[os.path.join(disps,i) for i in os.listdir(disps)]
        self.load_model(pt)
        
    def load_model(self,pt):
        data=torch.load(pt,map_location='cpu')
        state= data['state_dict']
        if 'module' in list(data['state_dict'].keys())[0].split('.',1)[0]:
            state={'.'.join(i.split('.')[1:]):j for i,j in state.items()}
        self.model=FADNet(batchNorm=False,lastRelu=True)
        self.model.load_state_dict(state)
        self.model.eval()
        
    @staticmethod    
    def load_data_imn(leftname, rightname):
        left=np.array(Image.open(leftname).convert('RGB'))
        right=np.array(Image.open(rightname).convert('RGB'))
        h, w, _ = left.shape
        
        img_left = np.zeros([3, h, w], 'float32')
        img_right = np.zeros([3, h, w], 'float32')
        for c in range(3):
            img_left[c, :, :] = (left[:, :, c] - np.mean(left[:, :, c])) / np.std(left[:, :, c])
            img_right[c, :, :] = (right[:, :, c] - np.mean(right[:, :, c])) / np.std(right[:, :, c])
            
        crop_height,crop_width=576,960
        bottom_pad = crop_height-h
        right_pad = crop_width-w
        img_left = np.lib.pad(img_left,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
        img_right = np.lib.pad(img_right,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
        return torch.from_numpy(img_left).float(), torch.from_numpy(img_right).float(), h, w
    
    def detect_(self,ix,draw=True):
        if ix>len(self.lefts_pt):
            return 
        
        left=Image.open(self.lefts_pt[ix])
        right=Image.open(self.rights_pt[ix])
        gt_disp=self.readPFM(self.disp_pt[ix])[0]
        left=self.transform(left)
        right=self.transform(right)
        input=torch.cat((left,right),0)
        new_input=input.unsqueeze(0)
        input_var = F.interpolate(new_input, (576, 960), mode='bilinear')
        start_time=time.time()
        
        with torch.no_grad():
            output=self.model(input_var)[-1]
            
        # print(output)
        inference_time=float('%.2f'%((time.time()-start_time)*1000))
                
        out=self.scale_disp(output,(output.size()[0], 540, 960))
        
        disp=out[0, 0, :, :]
        np_disp=disp.detach().numpy()
        
        if draw:
            self.plot_show(np_disp,np.max(gt_disp))
        
        mask = (gt_disp > 0) & (gt_disp < 192)
    
        epe = np.mean(np.abs(gt_disp[mask] - np_disp[mask]))
       
        epe=float("{:.4f}".format(epe))
        
        print('inference time: {}ms'.format(inference_time))
        print('epe: ',epe)
        
        return np_disp,epe,inference_time
    
    def detect(self,ix,draw=True):
        if ix>len(self.lefts_pt):
            return 
        
        gt_disp=self.readPFM(self.disp_pt[ix])[0]
        
        left,right,height,width=self.load_data_imn(self.lefts_pt[ix],self.rights_pt[ix])
        left=Variable(left, requires_grad = False)
        right=Variable(right, requires_grad = False)
        
        input=torch.cat((left,right),0)
        input_var=input.unsqueeze(0)
        
        start_time=time.time()
        
        with torch.no_grad():
            output=self.model(input_var)[-1]
            
        inference_time=float('%.2f'%((time.time()-start_time)*1000))

        out=output.squeeze(0)
        disp = out.detach().numpy()
        np_disp = disp[0, :height, :width]
        
        if draw:
            self.plot_show(np_disp,np.max(gt_disp))
      
        mask = (gt_disp > 0) & (gt_disp < 192)
        epe = np.mean(np.abs(gt_disp[mask] - np_disp[mask]))
        epe=float("{:.4f}".format(epe))
        
        print('inference time: {}ms'.format(inference_time))
        print('epe: ',epe)
        
        return np_disp,epe,inference_time
        
    # @staticmethod
    # def plot_show(data,max_disp,size=(10,10),cmap='rainbow',off=True):
    #     print('current color: {}'.format(cmap))
    #     plt.figure(figsize=size)
    #     plt.imshow(data,vmin=0,vmax=max_disp,cmap=cmap)
    #     plt.axis(not off)
    #     plt.show()
        
    @staticmethod
    def scale_disp(disp, output_size=(1, 540, 960)):
        i_w = disp.size()[-1]
        o_w = output_size[-1]

        m = nn.Upsample(size=(output_size[-2], output_size[-1]), mode="bilinear")
        trans_disp = m(disp)
        
        trans_disp = trans_disp * (o_w * 1.0 / i_w)
        return trans_disp


    @staticmethod
    def readPFM(file): 
        with open(file, "rb") as f:
            type = f.readline().decode('latin-1')
            if "PF" in type:
                channels = 3
            elif "Pf" in type:
                channels = 1
            else:
                return None,None,None
            line = f.readline().decode('latin-1')
            width, height = re.findall('\d+', line)
            width = int(width)
            height = int(height)
            line = f.readline().decode('latin-1')
            BigEndian = True
            if "-" in line:
                BigEndian = False
            samples = width * height * channels
            buffer = f.read(samples * 4)
            fmt= '>' if BigEndian else '<'
            fmt = fmt + str(samples) + "f"
            img = unpack(fmt, buffer)
            img = np.reshape(img, (height, width))
            img = np.flipud(img)
        return img, height, width

    def __len__(self):
        return len(self.lefts_pt)
    
    
if __name__=='__main__':
    detecter=sceceflow_detect(pt='models/best.pth')

    for i in range(len(detecter)):
        detecter.detect_(i,draw=False)
    # detecter.detect_(0,draw=False)
