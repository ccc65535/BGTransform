import torch
import sys

sys.path.append(r"../braindecode")

from braindecode.augmentation import FrequencyShift # type: ignore
from braindecode.augmentation import FTSurrogate    # type: ignore
from braindecode.augmentation import GaussianNoise  # type: ignore
from braindecode.augmentation import IdentityTransform  # type: ignore
from braindecode.augmentation import SignFlip   # type: ignore
from braindecode.augmentation import SmoothTimeMask # type: ignore
from braindecode.augmentation import TimeReverse    # type: ignore
from braindecode.augmentation import ChannelsShuffle    # type: ignore



def augment(trans_type,X,y,p=.8,srate=250):


    if trans_type=='SignalFlip':
        transform = SignFlip(
            probability=p,
        )

    elif trans_type=='TimeReverse':
        transform = TimeReverse(
            probability=p,
        )

    elif trans_type=='SmoothTimeMask':
        transform = SmoothTimeMask(
            probability=p,
            mask_len_samples=int(X.shape[-1]/2)
        )

    elif trans_type=='GaussianNoise':
        transform = GaussianNoise(
            probability=p,
        )

    elif trans_type=='FrequencyShift':
        transform = FrequencyShift(
            probability=p,
            sfreq=srate
        )

    elif trans_type=='FTSurrogate':
        transform = FTSurrogate(
            probability=p,
        )

    elif trans_type=='ChannelsShuffle':
        transform = ChannelsShuffle(
            probability=p,
            p_shuffle=0.8
        )

    else:
        assert('aug method not founded!')

    aug_X,aug_y = transform(X,y)
      

    return aug_X,aug_y