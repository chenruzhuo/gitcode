import os
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
from networks.FADNet import FADNet
from struct import unpack
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import warnings 
warnings.filterwarnings('ignore')

class kitti_detect:
    normalize={'mean': [0.485, 0.456, 0.406],'std': [0.229, 0.224, 0.225]}
    transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ]
        )
   
    def __init__(self,root='kitti/training',pt='models/fadnet.pth'):
        lefts=os.path.join(root,'image_2')
        rights=os.path.join(root,'image_3')
        disps=os.path.join(root,'disp_occ_0')
        self.lefts_pt=[os.path.join(lefts,i) for i in os.listdir(lefts)]
        self.rights_pt=[os.path.join(rights,i) for i in os.listdir(rights)]
        self.disp_pt=[os.path.join(disps,i) for i in os.listdir(disps)]
        self.load_model(pt)
        
    def load_model(self,pt):
        data=torch.load(pt,map_location=torch.device('cpu'))
        state= data['state_dict']
        if 'module' in list(data['state_dict'].keys())[0].split('.',1)[0]:
            state={'.'.join(i.split('.')[1:]):j for i,j in state.items()}
        self.model=FADNet(batchNorm=False,lastRelu=True)
        self.model.load_state_dict(state)
        self.model.eval()
        
    @staticmethod    
    def load_data_imn(leftname, rightname,crop=(384,1280)):
        left=np.array(Image.open(leftname).convert('RGB'))
        right=np.array(Image.open(rightname).convert('RGB'))
        h, w, _ = left.shape
        
        img_left = np.zeros([3, h, w], 'float32')
        img_right = np.zeros([3, h, w], 'float32')
        for c in range(3):
            img_left[c, :, :] = (left[:, :, c] - np.mean(left[:, :, c])) / np.std(left[:, :, c])
            img_right[c, :, :] = (right[:, :, c] - np.mean(right[:, :, c])) / np.std(right[:, :, c])
            
        bottom_pad = crop[0]-h
        right_pad = crop[1]-w
        img_left = np.lib.pad(img_left,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
        img_right = np.lib.pad(img_right,((0,0),(0,bottom_pad),(0,right_pad)),mode='constant',constant_values=0)
        return torch.from_numpy(img_left).float(), torch.from_numpy(img_right).float(), h, w
    
    def detect_(self,ix,draw=True):
        if ix>len(self.lefts_pt):
            return 
        
        if ix%2==0:
            gt_disp = Image.open(self.disp_pt[ix//2])
            gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256
        else:
            gt_disp=None
            
        imgL_o = np.array(Image.open(self.lefts_pt[ix]).convert('RGB'))
        imgR_o = np.array(Image.open(self.rights_pt[ix]).convert('RGB'))

        imgL = self.transform(imgL_o).numpy()
        imgR = self.transform(imgR_o).numpy()

        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        top_pad = 384-imgL.shape[2]
        left_pad = 1280-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(left_pad, 0)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(left_pad, 0)),mode='constant',constant_values=0)
        
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))

        input_var=torch.cat((imgL, imgR), 1)
        print('imgL: ',imgL.shape)
        print('input_var: ',input_var.shape)
        
            
        start_time = time.time()
        output=self.model(input_var)[-1]
        inference_time=float('%.2f'%((time.time()-start_time)*1000))
        
        print('output: ',output.shape)
        out = torch.squeeze(output)      
        print('out: ',out.shape)
        
        pred_disp = out.data.numpy()
        top_pad   = 384-imgL_o.shape[0]
        left_pad  = 1280-imgL_o.shape[1]
        np_disp = pred_disp[top_pad:,left_pad:]
        print('pred_disp: ',pred_disp.shape)
        print('np_disp: ',np_disp.shape)
        
        if draw:
            self.plot_show(np_disp,np.max(gt_disp) if ix%2==0 else np.max(np_disp))
        
        epe=None
        d1=None
        if ix%2==0:
            mask = (gt_disp > 0) & (gt_disp < 192)
            epe = np.mean(np.abs(gt_disp[mask] - np_disp[mask]))
            epe=float("{:.4f}".format(epe))
            
            a=(np.abs(gt_disp[mask]-np_disp[mask])<3)
            b=(np.abs(gt_disp[mask]-np_disp[mask])/gt_disp[mask]<0.05)
            correct=(a|b).sum()
            d1='%.2f'%((1-correct/gt_disp[mask].size)*100)+'%'
                        
        print('inference time: {}ms'.format(inference_time))
        print('epe: ',epe)
        print('d1: ',d1)
        
        return np_disp,epe,d1,inference_time
    
    def detect(self,ix,draw=True):
        if ix>len(self.lefts_pt):
            return 
        
        if ix%2==0:
            gt_disp = Image.open(self.disp_pt[ix//2])
            gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256
        else:
            gt_disp=None
            
                    
        left,right,height,width=self.load_data_imn(self.lefts_pt[ix],self.rights_pt[ix])
        left=Variable(left, requires_grad = False)
        right=Variable(right, requires_grad = False)
        
        input=torch.cat((left,right),0)
        input_var=input.unsqueeze(0)
        
        print('imgL: ',left)
        print('input_var: ',input_var.shape)
        
        # print(input_var)
        
        start_time=time.time()
        
        with torch.no_grad():
            output=self.model(input_var)[-1]
            
        # print(output)   
        inference_time=float('%.2f'%((time.time()-start_time)*1000))

        out=output.squeeze(0)
        disp = out.detach().numpy()
        np_disp = disp[0, :height, :width]
        
        print('disp: ',disp.shape)
        print('np_disp: ',np_disp.shape)
        
        if draw:
            self.plot_show(np_disp,np.max(gt_disp) if ix%2==0 else np.max(np_disp))
        
        epe=None
        d1=None
        if ix%2==0:
            mask = (gt_disp > 0) & (gt_disp < 192)
            epe = np.mean(np.abs(gt_disp[mask] - np_disp[mask]))
            epe=float("{:.4f}".format(epe))
            
            a=(np.abs(gt_disp[mask]-np_disp[mask])<3)
            b=(np.abs(gt_disp[mask]-np_disp[mask])/gt_disp[mask]<0.05)
            correct=(a|b).sum()
            d1='%.2f'%((1-correct/gt_disp[mask].size)*100)+'%'
                        
        print('inference time: {}ms'.format(inference_time))
        print('epe: ',epe)
        print('d1: ',d1)
        
        return np_disp,epe,d1,inference_time
        
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

    def __len__(self):
        return len(self.lefts_pt)
    
    
if __name__=='__main__':
    # pt='trained/fadnet-pre4-KITTI2015-split/kitti.tar'
    pt='models/best.pth'
    detecter=kitti_detect(pt=pt)
    # for i in range(len(detecter)):
    #     detecter.detect_(i,draw=False)
    detecter.detect_(0,draw=False)
    os.system('python -m sceneflow_predict')