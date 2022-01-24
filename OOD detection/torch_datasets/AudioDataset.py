import pathlib
import torch
import torchaudio
import pandas as pd

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,raw_data_dir,window_length,sample_rate,mode,
                 normalize = True, only_speech = False, return_filename = False):
        
        self.return_filename = return_filename
        
        # whether to only sample speech files
        
        self.only_speech = only_speech
        
        # whether to normalize to zero mean and unit variance
        
        self.normalize = normalize
        
        # path to raw data folder
        
        self.raw_data_dir = pathlib.Path(raw_data_dir)
        
        """
        given the desired <window_length> and the <mode>, the
        corresponding csv file is read. Each row of the csv file has the
        following format:
        
        A/B,C,D
        
        Where:
        - A is the dataset name (0_AUDIOSET,1_COUGH,...)
        - B is the file name
        - C is the start time of the window in seconds
        - D is the end time of the window in seconds
        """
        
        csv_path = pathlib.Path(self.raw_data_dir.parent,
                                str(window_length).replace('.','-') + 's',
                                'data_'+mode+'.csv')
        self.metadata = pd.read_csv(csv_path,header = None)
        
        # filter out non-speech files if necessary
        
        if only_speech:
            where = self.metadata[0].str.contains('0_LIBRISPEECH')
            self.metadata = self.metadata[where]
        else:
            labels = self.metadata[0].str.slice(0,1).astype(int)
            num_cough = sum(labels == 0)
            num_speech = sum(labels == 1)
            weights = 1. / torch.tensor([num_speech,num_cough])
            self.sample_weights = weights[torch.tensor(labels.to_list())]
        
        # what sampling rate to resample audio to
        
        self.new_sr = sample_rate
    
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self,idx):
        
        """
        load audio file. Since the torchaudio.load() function requires
        the <num_frames> and <offset> parameters to load a window of the
        audio file, then we first need to compute the location of the
        starting sample of the window and the length of the window using
        the sample rate. This sample rate is obtained using the
        torchaudio.info() function
        """
        
        path = str(self.raw_data_dir) + '/' + str(self.metadata.iloc[idx,0])
        start_sec = self.metadata.iloc[idx,1]
        end_sec = self.metadata.iloc[idx,2]

        path=str(path)
        data,sr=torchaudio.load(filepath = path) #find out original sample rate

        # sr = torchaudio.info(filepath = str(path))[0].rate
        start = round(sr * start_sec)
        length = round(sr * (end_sec - start_sec))
        x = torchaudio.load(filepath = path,frame_offset = start,num_frames = length)[0]
    
        x = torchaudio.transforms.Resample(sr,self.new_sr)(x)#transform the sr
        x = torch.mean(x,dim=0,keepdim=True)
        
        if not self.only_speech:
            
            label = int(self.metadata.iloc[idx,0][0])
            

            return x,label
        
        else:
            return x
        
        if self.normalize:
            x = (x - x.mean()) / x.std()
        
        # if not self.only_speech:
            
        #     # get audio signal label, 0 for speech, and 1 for cough
            
        #     label = int(self.metadata.iloc[idx,0][0])
            
        #     if self.return_filename:
        #         return (x,label,path.stem)
        #     else:
        #         return x,label
        
        # else:
        #     return x
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import sounddevice as sd
    import random
    
    raw_data_dir = '../data/raw'
    window_length = 1.5 # seconds
    sample_rate = 16000
    mode = 'test'
    dataset = AudioDataset(raw_data_dir,window_length,sample_rate,mode,
                           only_speech = False)
    
    # idx = random.randint(0,len(dataset))
    # x,label = dataset[idx]
    x = dataset[0]
    
    # compute log Mel spectrogram
    
    # log = torchaudio.transforms.AmplitudeToDB()
    # mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
    #                                                 n_fft = 1024,
    #                                                 n_mels = 64,
    #                                                 hop_length = 10)
    # log_mel_spec = log(mel_spec(x))
    
    # print('\nShowing signal for:\n{}'.format(dataset.paths[idx]))
    # plt.plot(x.numpy())
    # sd.play(x.numpy(),sample_rate)
