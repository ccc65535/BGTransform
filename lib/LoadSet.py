import mat73
import mne
import numpy as np
import scipy.io as sio
from scipy.signal import sosfiltfilt, cheby2, cheb2ord, cheby1, cheb1ord
from lib.Preprocess import *

def load_SSVEP_Benchmark(root_dir,channels):
    sampling_rate = 250  # Sampling rate
    sample_length = 1500
    total_channels = 64  # Determine total number of channel
    totalcharacter = 40
    totalblock = 6
    totalsubject = 35
    freq = []

    AllData = np.zeros(
        [len(channels), sample_length, totalcharacter, totalblock, totalsubject],dtype=np.float32)  # initializing


    ## Forming bandpass filters
    # High cut off frequencies for the bandpass filters (90 Hz for all)
    # high_cutoff = 90
    # # Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    # low_cutoff = 8
    filter_order = 4  # Filter Order of bandpass filters
    PassBandRipple_val = 1

    wp, ws=[8, 90],[6, 95]
    sos = cheby1(filter_order, PassBandRipple_val, wp, btype='bandpass', output='sos', fs=sampling_rate)

    ## Filtering
    for subject in range(totalsubject):
        s = subject + 1
        ss = str(s)

        # if s < 10:
        #     ss = '0' + str(s)
        # else:
        #     ss = str(s)

        # load data
        nameofdata = root_dir + 'S' + ss + '.mat'
        data = sio.loadmat(nameofdata)  # Loading the subject data
        sub_data = data['data'][channels]



        AllData[:, :, :, :, subject] = sosfiltfilt(sos,sub_data,axis=1)


    return AllData


def load_Nak(root_dir,down_srate):
    onset=38
    sampling_rate = 256  # Sampling rate
    sample_length = 1000
    total_channels = 8  # Determine total number of channel
    totalcharacter = 12
    totalblock = 15
    totalsubject = 10
    freq = []

    AllData = np.zeros(
        [total_channels, sample_length, totalcharacter, totalblock, totalsubject],dtype=np.float32)  # initializing


    ## Forming bandpass filters
    # High cut off frequencies for the bandpass filters (90 Hz for all)
    # high_cutoff = 90
    # # Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
    # low_cutoff = 8
    filter_order = 4  # Filter Order of bandpass filters
    PassBandRipple_val = 1

    wp, ws=[8, 90],[6, 95]
    sos = cheby1(filter_order, PassBandRipple_val, wp, btype='bandpass', output='sos', fs=sampling_rate)

    ## Filtering
    for subject in range(totalsubject):
        s = subject + 1
        ss = str(s)

        # if s < 10:
        #     ss = '0' + str(s)
        # else:
        #     ss = str(s)
        
        # load data
        nameofdata = root_dir + 'S' + ss + '.mat'
        data = sio.loadmat(nameofdata)  # Loading the subject data
        sub_data = data['eeg']
        sub_data=sub_data[:,:,onset:,:]
        sub_data=resample(sub_data,sampling_rate,down_srate,axis=2).transpose(1,2,0,3)
        sub_data=sub_data[:,:sample_length,:,:]




        AllData[:, :, :, :, subject] = sosfiltfilt(sos,sub_data,axis=1)


    return AllData


### Processing demo from 
### https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller/blob/main/Python/Load_Won2021dataset.ipynb

def extractEpoch3D(data, event, srate, baseline, frame, opt_keep_baseline):
    # extract epoch from 2D data into 3D [ch x time x trial]
    # input: event, baseline, frame
    # extract epoch = baseline[0] to frame[2]

    # for memory pre-allocation
    if opt_keep_baseline == True:
        begin_tmp = int(np.floor(baseline[0]/1000*srate))
        end_tmp = int(begin_tmp+np.floor(frame[1]-baseline[0])/1000*srate)
    else:
        begin_tmp = int(np.floor(frame[0]/1000*srate))
        end_tmp = int(begin_tmp+np.floor(frame[1]-frame[0])/1000*srate)
    
    epoch3D = np.zeros((data.shape[0], end_tmp-begin_tmp, len(event)))
    nth_event = 0

    for i in event:
        if opt_keep_baseline == True:
            begin_id = int(i + np.floor(baseline[0]/1000 * srate))
            end_id = int(begin_id + np.floor((frame[1]-baseline[0])/1000*srate))
        else:
            begin_id = int(i + np.floor(frame[0]/1000 * srate))
            end_id = int(begin_id + np.floor((frame[1]-frame[0])/1000*srate))
        
        tmp_data = data[:, begin_id:end_id]

        begin_base = int(np.floor(baseline[0]/1000 * srate))
        end_base = int(begin_base + np.floor(np.diff(baseline)/1000 * srate)-1)
        base = np.mean(tmp_data[:, begin_base:end_base], axis=1)

        rmbase_data = tmp_data - base[:, np.newaxis]
        epoch3D[:, :, nth_event] = rmbase_data
        nth_event = nth_event + 1

    return epoch3D

def load_P3(root_dir):

    # total_channels=32
    down_srate=100
    # nclass=2
    # sample_length=int(600*srate)
    totalsubject=55

    # AllData = np.zeros(total_channels, sample_length, nclass, , totalsubject],dtype=np.float32)  # initializing

    target_data,nontarget_data=[],[]
    ## Filtering
    for subject in range(totalsubject):
        s = subject + 1
        # ss = str(s)

        if s < 10:
            ss = '0' + str(s)
        else:
            ss = str(s)
        
        # load data
        nameofdata = root_dir + 's' + ss + '.mat'
        EEG = mat73.loadmat(nameofdata)  # Loading the subject data

        baseline = [-200, 0] # in ms
        frame = [0, 600] # in ms
        for n_calib in range(len(EEG['train'])):
            data = np.asarray(EEG['train'][n_calib]['data'])
            srate = EEG['train'][n_calib]['srate']
            data = butter_bandpass_filter(data, 0.5, 10, srate, 4)
            markers = EEG['train'][n_calib]['markers_target']

            targetID = np.where(markers==1)[0]
            nontargetID = np.where(markers==2)[0]

            tmp_targetEEG = extractEpoch3D(data, targetID, srate, baseline, frame, False)
            tmp_nontargetEEG = extractEpoch3D(data, nontargetID, srate, baseline, frame, False)
            if n_calib == 0:
                targetEEG = tmp_targetEEG
                nontargetEEG = tmp_nontargetEEG
            else:
                targetEEG = np.dstack((targetEEG, tmp_targetEEG))
                nontargetEEG = np.dstack((nontargetEEG, tmp_nontargetEEG))

        target_data.append(targetEEG)
        nontarget_data.append(nontargetEEG)

    target_data=np.array(target_data)
    nontarget_data=np.array(nontarget_data)

    tar_y=np.ones((target_data.shape[0],target_data.shape[-1]))
    non_y=np.zeros((nontarget_data.shape[0],nontarget_data.shape[-1]))

    eeg=np.concatenate((target_data,nontarget_data),axis=-1)
    y=np.concatenate((tar_y,non_y),axis=-1)

    eeg = resample(eeg, srate, down_srate, axis=2)

    return eeg,y


from lib.BrainInvader import BrainInvaders2012
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_BrainInv(chns=[ 'C3', 'Cz', 'C4',  'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']):

    # total_channels=32
    down_srate=100
    # nclass=2
    # sample_length=int(600*srate)
    totalsubject=43

    dataset = BrainInvaders2012(Training=True)

    eeg,label,sub=[],[],[]
    # get the data from subject of interest
    for subject in dataset.subject_list:

        data = dataset._get_single_subject_data(subject)
        raw = data['session_1']['run_training']

        srate=raw.info['sfreq']
        # chnames=raw.info['ch_names']



        # filter data and resample
        fmin = 1
        fmax = 24
        raw.filter(fmin, fmax, verbose=False)
        

        # detect the events and cut the signal into epochs
        events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
        event_id = {'NonTarget': 1, 'Target': 2}
        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0, verbose=False, preload=True)
        epochs.pick_types(eeg=True)
        epochs.pick_channels(chns)

        # get trials and labels
        X = epochs.get_data()

        y = events[:, -1]
        y = LabelEncoder().fit_transform(y)
        
        eeg.append(X)
        label.append(y)
        sub.append(np.ones(len(y))*subject)

    eeg=np.concatenate(eeg,axis=0)
    label=np.concatenate(label,axis=0)
    sub=np.concatenate(sub,axis=0)

    eeg = resample(eeg, srate, down_srate, axis=-1)
    eeg=eeg[:,:,int(0.2*down_srate):]

    meta=pd.DataFrame(zip(label,sub),columns=['class','subject'])
    

    return eeg,meta

def load_cnt(file, srate=1000):
    raw = mne.io.read_raw_cnt(file)
    events = raw.annotations
    eeg = raw[:][0]
    event_type = [int(type) for type in events.description]
    event_latency = [int(lat * srate) for lat in events.onset]

    ch_dict = {}
    i = 0
    for ch in raw.ch_names:
        ch_dict[ch] = i
        i += 1
    return eeg, event_type, event_latency, ch_dict

def load_eeglab(file):


    # Read EEGLAB epochs with data in .fdt file
    epochs = mne.io.read_raw_eeglab(file)
    
    # Access the data
    # print(epochs.get_data())

    return epochs.get_data()


# eeg,meta=load_BrainInv()
