import torch
import torchaudio
import time

from models.disc import Disc
from torch_datasets.AudioDataset import AudioDataset
from torch_datasets.mydataset import testdataset2

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
print('\nTraining using window length of {} seconds...'.format(window_length))

# what sampling frequency to resample windows to
sr = 16000 # Hz

# initialize loss function (negative log-likelihood function for
# Bernoulli distribution)
loss_func = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64).to(device)

# initialize discriminator network
FENet_param_path = 'parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)

# initialize optimizer. Must put net parameters on GPU before this
# step
optimizer = torch.optim.Adam(params = net.parameters(),
                             lr = 0.0003)

# number of epoch to train and validate for
num_epochs = 5

# where to save net parameters in a .pt file
pt_filename = '{}s_{}Hz_{}epochs.pt'.format(str(window_length).replace('.','-'),
                                            sr,num_epochs)
param_path = 'parameters/disc/' + pt_filename

# initialize training and validation dataloaders
dataloaders = {}
dl_config = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
batch_size = 64
                             
for mode in ['train','val']:
    
    dataset = testdataset2(raw_data_dir,window_length,sr,mode,
                            only_speech = False)
    
    dataloaders[mode] = torch.utils.data.DataLoader(
                               dataset = dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               num_workers=0)

# record the best validation loss across epochs
best_val_loss = 1e10

print("train_loader的batch数量为",len(dataloaders['train']))

# train or validate on a batch
def run_batch(mode,batch,transforms,net,loss_func,optimizer,device):
    
    x,labels = batch
    log,mel_spec = transforms
    x = x.to(device)
    labels = labels.to(device).type_as(x) # needed for NLL
    
    with torch.set_grad_enabled(mode == 'train'):
    
        # compute log Mel spectrogram
        
        log_mel_spec = log(mel_spec(x))
        
        # logits must have same shape as labels
        
        logits = net(log_mel_spec).squeeze(dim = 1)
        
        # compute negative log-likelihood (NLL) using logits
        
        NLL = loss_func(logits,labels)
        
        if mode == 'train':
        
            # compute gradients of NLL with respect to parameters
            
            NLL.backward()
            
            # update parameters using these gradients. Minimizing the negative
            # log-likelihood is equivalent to maximizing the log-likelihood
            
            optimizer.step()
            
            # zero the accumulated parameter gradients
            
            optimizer.zero_grad()
    
    # record predictions. since sigmoid(0) = 0.5, then negative values
    # correspond to class 0 and positive values correspond to class 1
    
    preds = logits > 0
    
    # record correct predictions
    
    true_preds = torch.sum(preds == labels)
    
    return NLL.item(),true_preds.item()

# train or validate over all batches
def run_epoch(mode,dataloader,transforms,net,loss_func,optimizer,device):
    
    if mode == 'train':
        print('Training...')
        net.train()
    else:
        print('\nValidating...')
        net.eval()
    
    # to compute average negative log-likelihood (NLL) per sample
    
    total_loss = 0
    
    # to compute training accuracy per epoch
    
    total_true_preds = 0
    
    for i,batch in enumerate(dataloader):
        
        # track progress
        
        print('\rProgress: {:.2f}%'.format((i+1)/len(dataloader)*100),
              end='',flush=True)
        
        # train or validate over the batch
        
        loss,true_preds = run_batch(mode,batch,transforms,net,loss_func,
                                    optimizer,device)
        
        # record running statistics
        
        total_loss += loss
        total_true_preds += true_preds
        
    loss_per_sample = total_loss / len(dataloader.dataset)
    acc = total_true_preds / len(dataloader.dataset)
    
    return loss_per_sample,acc

if __name__ == '__main__':
    
    start = time.time()
    
    for epoch in range(num_epochs):
        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        
        epoch_start = time.time()
    
        train_loss,train_acc = run_epoch('train',dataloaders['train'],
                                         (log,mel_spec),
                                         net,
                                         loss_func,
                                         optimizer,
                                         device)
        
        print('\nAverage Training loss: {:.4f}'.format(train_loss))
        print('Training Accuracy: {:.2f}%'.format(train_acc*100))
        
        val_loss,val_acc = run_epoch('val',dataloaders['val'],
                                     (log,mel_spec),
                                     net,
                                     loss_func,
                                     optimizer,
                                     device)
        
        print('\nAverage Validation Loss: {:.4f}'.format(val_loss))
        print('Validation Accuracy: {:.2f}%'.format(val_acc*100))
        
        epoch_end = time.time()
        
        epoch_time = time.strftime("%H:%M:%S",time.gmtime(epoch_end-epoch_start))
        
        print('\nEpoch Elapsed Time (HH:MM:SS): ' + epoch_time)
        
        if val_loss < best_val_loss:
            print('Saving checkpoint...')
            best_val_loss = val_loss
            torch.save(net.state_dict(),param_path)
    
    end = time.time()
    total_time = time.strftime("%H:%M:%S",time.gmtime(end-start))
    print('\nTotal time elapsed (HH:MM:SS): ' + total_time)
    print('Best validation loss: {:.2f}%'.format(best_val_loss))
