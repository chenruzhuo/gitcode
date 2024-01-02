import torch
from networks.FADNet import FADNet
import warnings
warnings.filterwarnings('ignore')

# pt='models/best.pth'
pt='trained/fadnet-pre4-KITTI2015-split/kitti.tar'
data=torch.load(pt,map_location='cpu')
state= data['state_dict']
if 'module' in list(data['state_dict'].keys())[0].split('.',1)[0]:
    state={'.'.join(i.split('.')[1:]):j for i,j in state.items()}
model=FADNet(batchNorm=False, lastRelu=True)
model.load_state_dict(state)
model.eval()


batch_size=1
input=(6,384,1280)
dummy_input=torch.randn(batch_size,*input)
input_names=['input']
output_names=['output1','output2']
torch.onnx.export(model,dummy_input,
                  'onnx/kitti4.onnx',
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  do_constant_folding=True,
                  opset_version=17,
                  dynamic_axes={'input': {0: 'batch', 1: 'channel', 2: 'height', 3: 'width'},
		 				'output1': {0: 'batch', 1: 'channel', 2: 'height', 3: 'width'},
                        'output2': {0: 'batch', 1: 'channel', 2: 'height', 3: 'width'},
                    }) 