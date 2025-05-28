import scipy.signal as signal
import mne
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2 * freq0 / srate
    wn2 = 2 * freq1 / srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def bandpass_butt(sig, lpass, hpass, fs, axis=-1):
    # wn1=2*freq0/srate
    # wn2=2*freq1/srate
    # % 通带的截止频率为2.75 hz - -75hz, 有纹波
    Wp = np.array([lpass, hpass]) / (fs / 2)
    #  % 阻带的截止频率
    Ws = np.array([(lpass - 2), (hpass + 10)]) / (fs / 2)
    # % 通带允许最大衰减为 db
    alpha_pass = 1

    #  % 阻带允许最小衰减为  db
    alpha_stop = 5
    # % 获取阶数和截止频率
    N, Wn = signal.buttord(Wp, Ws, alpha_pass, alpha_stop)

    b, a = signal.butter(N, Wn, 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def bandpass_cheby1(sig, lpass, hpass, fs, axis=-1):
    # wn1=2*freq0/srate
    # wn2=2*freq1/srate
    # % 通带的截止频率为2.75 hz - -75hz, 有纹波
    Wp = np.array([lpass, hpass]) / (fs / 2)
    #  % 阻带的截止频率
    Ws = np.array([(lpass - 2), (hpass + 5)]) / (fs / 2)
    # % 通带允许最大衰减为 db
    alpha_pass = 3

    #  % 阻带允许最小衰减为  db
    alpha_stop = 25
    # % 获取阶数和截止频率
    N, Wn = signal.cheb1ord(Wp, Ws, alpha_pass, alpha_stop)

    b, a = signal.cheby1(N, 0.5, Wn, 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def resample(sig, orginFreq, newFreq, axis=-1):
    return mne.filter.resample(sig, down=orginFreq, up=newFreq, npad='auto', axis=axis)


def bandpass_cheby11(sig, lpass, hpass, fs, axis=-1):
    # wn1=2*freq0/srate
    # wn2=2*freq1/srate
    # % 通带的截止频率为2.75 hz - -75hz, 有纹波
    Wp = np.array([lpass, hpass])
    #  % 阻带的截止频率
    Ws = np.array([(lpass - 4), (hpass + 10)])
    # % 通带允许最大衰减为 db
    alpha_pass = 1

    #  % 阻带允许最小衰减为  db
    alpha_stop = 5
    # % 获取阶数和截止频率
    N, Wn = signal.cheb1ord(Wp, Ws, alpha_pass, alpha_stop, fs=fs)

    b, a = signal.cheby1(N, 2, Wn, 'bandpass', fs=fs)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)

    return sig_new

def notch(sig,tar_freq,bw,fs,axis=-1):

    Q = tar_freq/bw
    # Design notch filter
    b, a = signal.iirnotch(tar_freq, Q, fs)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)

    return sig_new

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order, [low, high], btype='band')
  # demean before filtering
  meandat = np.mean(data, axis=1)
  data = data - meandat[:, np.newaxis]
  y = filtfilt(b, a, data)
  return y




