

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt

from torch import Tensor
import sys
from torch.utils.data import TensorDataset

import time
import os
import numpy as np

import PIL.Image
import sklearn.metrics

from vocparseclslabels import PascalVOC

from typing import Callable, Optional

from sklearn.metrics import average_precision_score


class dataset_voc(Dataset):
    def __init__(self, root_dir, trvaltest, transform=None):

        #TODO
        #read in pascal VOC dataset

        self.root_dir = root_dir
        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]

        pv=PascalVOC(self.root_dir)
        self.categories=pv.list_image_sets()

        if trvaltest==0:
            self.read_data(pv, 'train')
        if trvaltest==1:
            self.read_data(pv, 'val')
        if trvaltest==2:
            self.read_data(pv, 'trainval')


    def read_data(self, pv, trvaltest):
        for i, cat_name in enumerate(self.categories):
            for fn in pv.imgs_from_category_as_list(cat_name, trvaltest):
                if (fn not in self.imgfilenames):
                    self.imgfilenames.append(fn)
                    one_hot=[0] * len(self.categories)
                    one_hot[i]=1
                    self.labels.append(one_hot)
                else:
                    j=self.imgfilenames.index(fn)
                    self.labels[j][i]=1
        self.labels = np.asarray(self.labels)
        self.labels = torch.from_numpy(self.labels.astype('long'))

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        #TODO your code here
        file_path=os.path.join(self.root_dir, 'JPEGImages', self.imgfilenames[idx] + ".jpg")
        image = PIL.Image.open(file_path).convert('RGB')

        if self.transform:
          image = self.transform(image)

        label=self.labels[idx]

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample


def train_epoch(model,  trainloader,  criterion, device, optimizer ):

    #TODO
    #model.train() or model.eval() ?
    model.train()

    losses = []
    #print("Count to: ", len(trainloader))
    for batch_idx, data in enumerate(trainloader):
      #TODO
      #print(batch_idx)
      inputs=data['image']
      labels=data['label']

      inputs= inputs.to(device)

      optimizer.zero_grad()

      output = model(inputs)
      labels = labels.type_as(output)
      labels = labels.to(device)
      loss = criterion(output, labels)

      loss.backward()
      optimizer.step()

      losses.append(loss.item())
      if batch_idx%100==0:
        print('current mean of losses ',np.mean(losses))

    return np.mean(losses)



def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    #TODO
    #model.train() or model.eval() ?
    model.eval()

    curcount = 0
    accuracy = 0

    #concat_pred=[np.empty(shape=(0)) for _ in range(numcl)] #prediction scores for each class. each numpy array is a list of scores. one score per image
    #concat_labels=[np.empty(shape=(0)) for _ in range(numcl)] #labels scores for each class. each numpy array is a list of labels. one label per image
    concat_pred = np.empty((0,numcl))
    concat_labels = np.empty((0,numcl))
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader


    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):


          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ',batch_idx)

          inputs = data['image'].to(device)
          outputs = model(inputs)

          labels = data['label']

          labels = labels.type_as(outputs)
          loss = criterion(outputs, labels.to(device) )
          losses.append(loss.item())

          #this was an accuracy computation
          cpuout= outputs.to('cpu')
          #_, preds = torch.max(cpuout, 1)
          #labels = labels.float()
          #corrects = torch.sum(preds == labels.data)
          #accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + corrects.float()* ( curcount/ float(curcount+labels.shape[0]) )


          #TODO: collect scores, labels, filenames
          for i, d in enumerate(data):
              concat_pred=np.vstack((concat_pred,cpuout[i].cpu()))
              concat_labels=np.vstack((concat_labels,labels[i].cpu()))
              fnames.append(data['filename'][i])
          curcount+= labels.shape[0]
    for c in range(numcl):
      avgprecs[c]=average_precision_score(concat_labels[:,c], concat_pred[:,c])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)
    #print(avgloss)

    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)

    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights= model.state_dict()
      #TODO track current best performance measure and epoch
      best_measure=avgperfmeasure
      best_epoch=epoch
      #TODO save your scores
      best_scores={'epoch': best_epoch, 'weights': model.state_dict(), 'avgperfmeasure': best_measure,
                    'outputs': concat_pred, 'filenames': fnames, 'labels': concat_labels}

  torch.save(best_scores, "results_predictions")
  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs




class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__()
        self.loss=torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        loss=self.loss(input_,target)
        return loss





def runstuff():


    config = dict()

    config['use_gpu'] = True #True #TODO change this to True for training on the cluster, eh
    config['lr']=0.005
    config['batchsize_train'] = 16
    config['batchsize_val'] = 64
    config['maxnumepochs'] = 35

    config['scheduler_stepsize']=10
    config['scheduler_factor']=0.3



    # kind of a dataset property
    config['numcl']=20



    #data augmentations
    data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }


    #root_dir='./data/VOCdevkit/VOC2012/'
    root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/VOCdevkit/VOC2012/'
    #datasets
    image_datasets={}
    image_datasets['train']=dataset_voc(root_dir=root_dir,trvaltest=0, transform=data_transforms['train'])
    image_datasets['val']=dataset_voc(root_dir=root_dir,trvaltest=1, transform=data_transforms['val'])

    #dataloaders
    #TODO use num_workers=1
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], shuffle=False, num_workers=1)


    #device
    if True == config['use_gpu']:
      device= torch.device('cuda:0')

    else:
      device= torch.device('cpu')

    #model
    #TODO
    model = models.resnet18(pretrained=True)#pretrained resnet18
    #overwrite last linear layer
    num_ftrs = model.fc.in_features
    print(num_ftrs)
    model.fc = nn.Linear(num_ftrs, config['numcl'])

    model = model.to(device)

    lossfct = yourloss().to(device)

    #TODO
    # Observe that all parameters are being optimized
    someoptimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    # Decay LR by a factor of 0.3 every X epochs
    #TODO
    #somelr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=someoptimizer, mode='min', factor=.3, verbose=True)
    somelr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=someoptimizer, gamma=.3, last_epoch=-1, verbose=True)

    best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )



###########
# for part2
###########
'''


def setbyname2(targetmodel,name,value):

    def iteratset(obj,components,value,nametail=[]):

      if not hasattr(obj,components[0]):
        return False
      elif len(components)==1:
        if not hasattr(obj,components[0]):
          print('object has not the component:',components[0])
          print('nametail:',nametail)
          exit()
        setattr(obj,components[0],value)
        #print('found!!', components[0])
        #exit()
        return True
      else:
        nextobj=getattr(obj,components[0])

        newtail = nametail
        newtail.append(components[0])
        #print('components ',components, nametail, newtail)
        #print(type(obj),type(nextobj))

        return iteratset(nextobj,components[1:],value, nametail= newtail)

    components=name.split('.')
    success=iteratset(targetmodel,components,value, nametail=[])
    return success



class wsconv2(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation = 1 , groups =1 , bias = None, eps=1e-12 ):
    super(wsconv2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    self.eps=eps

  def forward(self,x):
    #torch.nn.functional.conv2d documentation tells about weight shapes
    pass


def bntoWSconverter(model):

  #either you modify model in place
  #or you create a copy of it e.g. using copy.deepcopy(...)
  # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/17

  lastwasconv2= False
  for nm,module in model.named_modules():
    #print(nm)

    if isinstance(module, nn.Conv2d):
      #replace, get std
      lastwasconv2= True

      usedeps= 1e-12 # use 1e-12 if you add it to a variance term, and 1e-6 if you add it to a standard deviation term

      #TODO
      # put in here your wsconv2, dont forget to copy convolution weight and, if exists, the convolution bias into your wsconv2

      setbyname2(model,nm,newconv)

    elif isinstance(module,nn.BatchNorm2d):

      if False == lastwasconv2:
        print('got disconnected batchnorm??')
        exit()


      print('got one', nm)

      #TODO
      # you will need here data computed from the preceding nn.Conv2d instance which came along your way

      #delete
      lastwasconv2= False

    else:
      lastwasconv2= False




#preprocessing: https://pytorch.org/docs/master/torchvision/models.html
#transforms: https://pytorch.org/docs/master/torchvision/transforms.html
#grey images, best dealt before transform
# at first just smaller side to 224, then 224 random crop or centercrop(224)
#can do transforms yourself: PIL -> numpy -> your work -> PIL -> ToTensor()

class dataset_imagenetvalpart(Dataset):
  def __init__(self, root_dir, xmllabeldir, synsetfile, maxnum, transform=None):

    """
    Args:

        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    self.root_dir = root_dir
    self.xmllabeldir=xmllabeldir
    self.transform = transform
    self.imgfilenames=[]
    self.labels=[]
    self.ending=".JPEG"

    self.clsdict=get_classes()


    indicestosynsets,self.synsetstoindices,synsetstoclassdescr=parsesynsetwords(synsetfile)


    for root, dirs, files in os.walk(self.root_dir):
       for ct,name in enumerate(files):
          nm=os.path.join(root, name)
          #print(nm)
          if (maxnum >0) and ct>= (maxnum):
            break
          self.imgfilenames.append(nm)
          label,firstname=parseclasslabel(self.filenametoxml(nm) ,self.synsetstoindices)
          self.labels.append(label)


  def filenametoxml(self,fn):
    f=os.path.basename(fn)

    if not f.endswith(self.ending):
      print('not f.endswith(self.ending)')
      exit()

    f=f[:-len(self.ending)]+'.xml'
    f=os.path.join(self.xmllabeldir,f)

    return f


  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):
    image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

    label=self.labels[idx]

    if self.transform:
      image = self.transform(image)

    #print(image.size())

    sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

    return sample




def comparetwomodeloutputs(model1, model2, dataloader, device):

    model1.eval()
    model2.eval()

    curcount = 0
    avgdiff = 0

    with torch.no_grad():
      for batch_idx, data in enumerate(dataloader):


          if (batch_idx%100==0) and (batch_idx>=100):
              print('at val batchindex: ', batch_idx)

          inputs = data['image'].to(device)
          outputs1 = model1(inputs)
          outputs2 = model2(inputs)

          diff=torch.mean(torch.abs((outputs1-outputs2).flatten()))

          labels = data['label']
          print('diff',diff.item())
          avgdiff = avgdiff*( curcount/ float(curcount+labels.shape[0]) ) + diff.item()* ( labels.shape[0]/ float(curcount+labels.shape[0]) )


          curcount+= labels.shape[0]

    return avgdiff


#routine to test that your copied model at evaluation time works as intended
def test_WSconversion():


  config = dict()

  #config['use_gpu'] = True
  #config['lr']=0.008 #0.005
  #config['batchsize_train'] = 2
  #config['batchsize_val'] = 64

  #data augmentations
  data_transforms = {
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          #transforms.RandomHorizontalFlip(), # we want no randomness here :)
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/imagenet300/'
  xmllabeldir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/val/'
  synsetfile='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/students/synset_words.txt'

  dset= dataset_imagenetvalpart(root_dir, xmllabeldir, synsetfile, maxnum=64, transform=data_transforms['val'])
  dataloader =  torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False) #, num_workers=1)

  import copy
  device=torch.device('cpu')
  #model
  model = models.resnet18(pretrained=True)
  model2 = copy.deepcopy(model.to('cpu'))

  ####################
  # assumes it changes the model in-place, use model2= bntoWSconverter(model) if your routine instead modifies a copy of model and returns it
  ######################
  bntoWSconverter(model2)

  model = model.to(device)
  model2 = model2.to(device)

  avgdiff = comparetwomodeloutputs(model, model2, dataloader, device)



  print('model checking averaged difference', avgdiff )  # order 1e-3 is okay, 1e-2 is still okay.




'''




if __name__=='__main__':

  runstuff()
  #concat_pred=[np.empty(shape=(0)) for _ in range(12)]
  """concat_pred = np.empty((0,12))
  print(concat_pred)
  a=np.ones((12))
  b=np.ones((12))
  print(np.shape(a),np.shape(concat_pred))
  concat_pred=np.vstack((concat_pred,a))
  print(concat_pred)
  concat_pred=np.vstack((concat_pred,b))
  print(concat_pred)"""
  #root_dir='./data/VOCdevkit/VOC2012/'

  """data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }
  image_datasets={}
  image_datasets['train']=dataset_voc(root_dir=root_dir,trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=dataset_voc(root_dir=root_dir,trvaltest=1, transform=data_transforms['val'])

  print(image_datasets['train'].__getitem__(0))
  print(image_datasets['val'].__getitem__(0))"""
