import torch
import torchaudio
import csv
import torch.utils.data as Data
import pandas as pd
from torch_datasets.mydataset import testdataset2

from models.disc import Disc
# from torch_datasets.AudioDataset import AudioDataset

import warnings
# supresses torchaudio warnings. Should not be used in development
warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(42)

# to put tensors on GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# where the raw data is
raw_data_dir = 'data/raw'

# what window length to use
window_length = 1.5 # seconds
print('\nTesting window length of {} seconds...'.format(window_length))

# what sampling frequency to resample windows to
sr = 16000 # Hz

# initialize loss function (negative log-likelihood function for
# Bernoulli distribution)
loss_func = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().cuda()

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64).cuda()

# initialize discriminator network and load pre-trained parameters
FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).cuda()
# number of epochs that the net to be tested was trained for. This must
# match the num_epochs variable in train.py
num_epochs = 5
param_path = 'parameters/disc/{}s_{}Hz_{}epochs.pt'.format(str(window_length).replace('.','-'),
                                                           sr,num_epochs)
net.load_state_dict(torch.load(param_path,map_location = device))

# initialize training and validation dataloaders
dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
batch_size = 64


###############################
#editing part

test_data=testdataset2(raw_data_dir,window_length,sr,'test')



##############################

    


dataloader = torch.utils.data.DataLoader(
                           dataset = test_data,
                           batch_size = batch_size,
                           shuffle = True,
                           drop_last = False,
                           num_workers=0)

print("testloader的batch数量为：",len(dataloader))

@torch.no_grad()
def run_batch(mode,x,labels,transforms,net,loss_func):
    
    log, mel_spec = transforms
    
    log_mel_spec = log(mel_spec(x))      
    logits = net(log_mel_spec).squeeze(dim = 1)
    
    NLL = loss_func(logits,labels)
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    preds = logits > 0
    
    # true positives
    TP = torch.sum(torch.logical_and(preds == 1,labels == 1))
    # false positives
    FP = torch.sum(torch.logical_and(preds == 1,labels == 0))
    # true negatives
    TN = torch.sum(torch.logical_and(preds == 0,labels == 0))
    # false negatives
    FN = torch.sum(torch.logical_and(preds == 0,labels == 1))
    
    # confusion matrix
    CM = torch.tensor([[TN,FP],[FN,TP]])
    print("正在计算CM")
    return NLL.item(),CM

def run_epoch(mode,dataloader,transforms,disc_net,disc_loss_func,device):
    
    log, mel_spec = transforms
    
    disc_net.eval()
    
    total_disc_loss = 0
    total_CM = 0
    
    for i,(x,labels) in enumerate(dataloader):
        
        print('\rProgress: {:.2f}%'.format((i+1)/len(dataloader)*100),
              end='',flush=True)
        
        x = x.cuda()
        labels = labels.cuda().type_as(x)
        
        disc_loss,CM = run_batch(mode,x,labels,(log,mel_spec),disc_net,
                                 disc_loss_func)
        
        total_disc_loss += disc_loss
        total_CM += CM
    
    mean_disc_loss = total_disc_loss * dataloader.batch_size / len(dataloader.dataset)
    
    return mean_disc_loss, total_CM

def save_metrics(CM,disc_loss,metrics_path):
    
    fp = open(metrics_path,mode='w')
    csv_writer = csv.writer(fp,delimiter=',',lineterminator='\n')
    
    csv_writer.writerow(['Discriminator Loss','{:.4f}'.format(disc_loss)])
    csv_writer.writerow(['Discriminator Stats'])
    
    TP = CM[1,1]
    FP = CM[0,1]
    TN = CM[0,0]
    FN = CM[1,0]
    sensitivity = TP/(TP+FN+1e-10) # true positive rate (TPR)
    csv_writer.writerow(['Sensitivity/Recall','{:.3f}'.format(sensitivity)])
    specificity = TN/(TN+FP+1e-10) # true negative rate (TNR)
    csv_writer.writerow(['Specificity','{:.3f}'.format(specificity)])
    accuracy = (TP+TN)/(TP+TN+FP+FN+1e-10)
    csv_writer.writerow(['Accuracy','{:.3f}'.format(accuracy)])
    balanced_accuracy = (sensitivity+specificity)/2
    csv_writer.writerow(['Balanced accuracy','{:.3f}'.format(balanced_accuracy)])
    
    # Matthews correlation coefficient
    
    MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5+1e-10)
    csv_writer.writerow(['Matthews correlation coefficient','{:.3f}'.format(MCC)])
        
    # positive predictive value (or precision)
    
    PPV = TP/(TP+FP+1e-10)
    csv_writer.writerow(['Precision/PPV','{:.3f}'.format(PPV)])
    
    # negative predictive value
    
    NPV = TN/(TN+FN+1e-10)
    csv_writer.writerow(['NPV','{:.3f}'.format(NPV)])
    
    # close csv file after writing
    
    fp.close()
    
    metrics = {'CM':CM,
               'sensitivity':sensitivity,
               'specificity':specificity,
               'acc':accuracy,
               'bal_acc':balanced_accuracy,
               'MCC':MCC,
               'precision':PPV,
               'NPV':NPV}
    
    return metrics

if __name__ == '__main__':
        
    # train for an epoch
    
    mean_disc_loss, total_CM = run_epoch('test',
                                         dataloader,
                                         (log,mel_spec),
                                         net,
                                         loss_func,
                                         device)
    
    metrics_path = 'results/test/{}s_{}Hz_{}epochs_小部分test.csv'.format(str(window_length).replace('.','-'),
                                                               sr,num_epochs)
    metrics = save_metrics(total_CM,mean_disc_loss,metrics_path)
