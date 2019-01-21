import numpy as np 
from PIL import Image
import scipy.io as sio
import random
from eval_segm import*
np.set_printoptions(suppress=True)


def evaluator(data_file,label_path):
    label = sio.loadmat(label_path)['clas1']
    label-=1
    label_pred = np.load(data_file)
    print(label_pred.shape)
    mix = np.zeros([16,16])
    mix_prob = np.zeros([16,16])
    np.set_printoptions(precision=1)
    for i in range(1076):
        for j in range(1024):
            if label[i][j]==16 or label_pred[i][j]==16:
                continue
            mix[label[i][j],label_pred[i][j]] = mix[label[i][j],label_pred[i][j]]+1.
    precision,recall = [],[]
    mix_p = np.sum(mix,0)
    mix_r = np.sum(mix,1)
    K= 0.
    s = 0.
    total = 0.
    for i in range(16):
        total = total+np.sum(mix[i,:])
        precision.append(mix[i][i]/mix_p[i])
        recall.append(mix[i][i]/mix_r[i])
        K = K+np.sum(mix[i,:])*np.sum(mix[:,i])
        s = s+mix[i][i]
    ka = (s/total-K/(s**2))/(1-K/(s**2))
    print('**************Evalutation Results**************')
    print("Acc:"+str(s/total))
    print("kappa:"+str(ka))
    
    

    freq_w_iu,w_iu_lst = frequency_weighted_IU(label_pred[:1076,:1024],label[:1076,:1024])
    print("FwIoU: "+str(freq_w_iu))
    print("Confusion matrix:")
    for i in range(16):
        mix_prob[i,:] = mix[i,:]*100/mix_r[i]
        print(mix_prob[i,:])


if __name__=='__main__':
    evaluator("CRPM_Net_local.npy","data/label.mat")
