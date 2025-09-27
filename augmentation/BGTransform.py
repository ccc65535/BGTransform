import numpy as np
import random

from sympy import false

from augmentation.base import augment


def transform(trials,func:str,srate=None):
    nt,nch,nsamp=trials.shape

    template=np.mean(trials,axis=0,keepdims=True).repeat(nt,0)
    bg=trials-template
    if srate is None:
        bg_t=augment(func,bg,np.ones(bg.shape[0]),p=1)
    else:
        bg_t=augment(func,bg,np.ones(bg.shape[0]),p=1,srate=srate)
    gen_trials=template+bg_t[0].numpy()

    return gen_trials
    

def block_mix(block,func,nt,srate=None):
    nTrials,nChan,nSamp=block.shape

    rnd_ind=[i for i in range(nTrials)]
    
    random.shuffle(rnd_ind)
    block=block[rnd_ind,:,:]


    build_block=np.zeros(block.shape)
    
    for ns in range(0,nTrials,nt):
        tr_ind=[i%nTrials for i in range(ns,ns+nt)]
        

        trials=block[tr_ind,:,:]


        gen_trials=transform(trials,func,srate)

        build_block[tr_ind,:,:]=gen_trials


    return build_block


def generate_data(X,y,func,nt=[2,2,2],srate=None):

    # X: all data for 1 subject
    labels=np.unique(y)

    gen_X=[]
    gen_y=[]

    # iter all label
    for y1 in labels:
        # get data with label y1
        ind=np.where(y==y1)
        block_X=X[ind]
        block_y=y[ind]

        for k in nt:

            gen_block=block_mix(block_X,func,k,srate)

            for i in range(gen_block.shape[0]):
                gen_X.append(gen_block[i])
                gen_y.append(y1)
                
    return np.array(gen_X),np.array(gen_y)

def signal_flip(X):
    return -X

def time_reverse(X):
    return np.flip(X,axis=-1)


def gaussian_noise(X,scale=0.5):
    
    noise = np.random.normal(loc=np.zeros(X.shape), scale=scale)
    return X+noise

