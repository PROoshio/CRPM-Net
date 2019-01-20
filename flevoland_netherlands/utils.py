import numpy as np 
import pickle
import random
import logging
from PIL import Image
import scipy.io as sio
import matplotlib as mpl 
import matplotlib.pyplot as plt 



def rot(image,label,k):
    for i in range(image.shape[-1]):
        image[:,:,i] = np.rot90(image[:,:,i],k)
    label = np.rot90(label,k)
    return image,label

def flip(image,label):
    for i in range(image.shape[-1]):
        image[:,:,i] = np.fliplr(image[:,:,i])
    label = np.fliplr(label)
    return image,label

def image_add_border(img,image_size):
    b_size = [(image_size[0]-img.shape[0])/2,(image_size[1]-img.shape[1])/2]
    img_new = np.zeros([image_size[0],image_size[1],img.shape[2]])
    img_new[b_size[0]:image_size[0]-b_size[0],b_size[1]:image_size[1]-b_size[1],:] = img
    img_new[:b_size[0],b_size[1]:image_size[1]-b_size[1],:] = np.flipud(img[:b_size[0],:,:])
    img_new[image_size[0]-b_size[0]:,b_size[1]:image_size[1]-b_size[1],:] = np.flipud(img[img.shape[0]-b_size[0]:,:,:])
    img_new[:,:b_size[1],:] = np.fliplr(img_new[:,b_size[1]:b_size[1]*2,:])
    img_new[:,image_size[1]-b_size[1]:,:] = np.fliplr(img_new[:,image_size[1]-2*b_size[1]:image_size[1]-b_size[1],:])
    return img_new


def get_logger(filename):
    logger = logging.getLogger('PROoshio')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def matReader(data_path):
    load_data_c = sio.loadmat(data_path+'_c.mat')#dict-numpy.ndarray
    load_data_l = sio.loadmat(data_path+'_l.mat')
    load_data_p = sio.loadmat(data_path+'_p.mat')
    row,col = 1079,1024
    channel_num = 18
    data_real = np.zeros([row,col,channel_num])
    data_imag = np.zeros([row,col,channel_num])
    step = 0
    for key,val in load_data_p.items():
        if key not in ['__header__','__version__','__globals__']:
            if key=="c11":
                data_real[:,:,0] = Z_score(load_data_c[key])
                data_imag[:,:,0] = np.zeros_like(data_real[:,:,0])+1e-3
                data_real[:,:,1] = Z_score(load_data_p[key])
                data_imag[:,:,1] = np.zeros_like(data_real[:,:,1])+1e-3
                data_real[:,:,2] = Z_score(load_data_l[key])
                data_imag[:,:,2] = np.zeros_like(data_real[:,:,2])+1e-3
            elif key=="c22":
                data_real[:,:,3] = Z_score(load_data_c[key])
                data_imag[:,:,3] = np.zeros_like(data_real[:,:,0])+1e-3
                data_real[:,:,4] = Z_score(load_data_p[key])
                data_imag[:,:,4] = np.zeros_like(data_real[:,:,1])+1e-3
                data_real[:,:,5] = Z_score(load_data_l[key])
                data_imag[:,:,5] = np.zeros_like(data_real[:,:,2])+1e-3
            elif key=="c33":
                data_real[:,:,6] = Z_score(load_data_c[key])
                data_imag[:,:,6] = np.zeros_like(data_real[:,:,0])+1e-3
                data_real[:,:,7] = Z_score(load_data_p[key])
                data_imag[:,:,7] = np.zeros_like(data_real[:,:,1])+1e-3
                data_real[:,:,8] = Z_score(load_data_l[key])
                data_imag[:,:,8] = np.zeros_like(data_real[:,:,2])+1e-3
            elif key=="c12_im":
                data_imag[:,:,9] = Z_score(load_data_c[key])
                data_imag[:,:,10] = Z_score(load_data_p[key])
                data_imag[:,:,11] = Z_score(load_data_l[key])
            elif key=="c12_re":
                data_real[:,:,9] = Z_score(load_data_c[key])
                data_real[:,:,10] = Z_score(load_data_p[key])
                data_real[:,:,11] = Z_score(load_data_l[key])
            elif key=="c13_im":
                data_imag[:,:,12] = Z_score(load_data_c[key])
                data_imag[:,:,13] = Z_score(load_data_p[key])
                data_imag[:,:,14] = Z_score(load_data_l[key])
            elif key=="c13_re":
                data_real[:,:,12] = Z_score(load_data_c[key])
                data_real[:,:,13] = Z_score(load_data_p[key])
                data_real[:,:,14] = Z_score(load_data_l[key])
            elif key=="c23_im":
                data_imag[:,:,15] = Z_score(load_data_c[key])
                data_imag[:,:,16] = Z_score(load_data_p[key])
                data_imag[:,:,17] = Z_score(load_data_l[key])
            elif key=="c23_re":
                data_real[:,:,15] = Z_score(load_data_c[key])
                data_real[:,:,16] = Z_score(load_data_p[key])
                data_real[:,:,17] = Z_score(load_data_l[key])
    return data_real,data_imag

def Z_score(data):
    # Z-score nomalization
    data = (data-data.mean())/data.std()
    return data

def image_scan(window_size,point,row,col,data,is_training=False):
    image = np.zeros((window_size+1,window_size+1,data.shape[-1]))
    for i in range(point[0]-window_size//2,point[0]+window_size//2+2):
        for j in range(point[1]-window_size//2,point[1]+window_size//2+2):
            if i>=0 and i<row and j>=0 and j<col: 
                image[i-point[0]+window_size//2,j-point[1]+window_size//2,:] = data[i,j,:]
            else:
                image[i-point[0]+window_size//2,j-point[1]+window_size//2,:] = np.zeros([data.shape[-1]])
    image_rot = image.copy()
    if is_training:
        k = random.randint(0,4)
        for i in range(image_rot.shape[-1]):
            image_rot[:,:,i] = np.rot90(image_rot[:,:,i],k)
        image_flip = image.copy()
        for i in range(image_flip.shape[-1]):
            image_flip[:,:,i] = np.fliplr(image_flip[:,:,i])
        return [image,image_rot,image_flip]
    else:
        return image

def get_image_data(data,point,label_path,window_size,row,col):
    label = sio.loadmat(label_path)['clas1']
    data_image = []
    for i in point:
        #data_image.append(image_add_border(image_scan(window_size,i,row,col,data,False),[10,10]))
        data_image.append(image_scan(window_size,i,row,col,data,False))
    return data_image



def get_random_sample(label,label_local):
    num = np.array([600,400,600,600,600,700,700,300,240,100,600,100,100,400,200,400])
    row,col = label.shape
    label_lst = [[] for i in range(16)]
    for i in range(row):
        for j in range(col):
            if label_local.item((i,j))==17:
                continue
            label_lst[label_local.item((i,j))-1].append((i,j))
    train_lst = []
    test_lst = []
    for i in range(16):
        print(num1[i]/float(len(label_lst[i])),len(label_lst[i]))
        random.shuffle(label_lst[i])
        train_lst.extend(label_lst[i][:num[i]])
        test_lst.extend(label_lst[i][-num[i]/10:])
    with open('train.plk','wb') as wr:
        pickle.dump(train_lst,wr)
    with open('test.plk','wb') as wr:
        pickle.dump(test_lst,wr)

def get_train_data(data_path,label_path,label_path_local,rate):
    train_data,train_label = [],[]
    test_data,test_label = [],[]
    label = sio.loadmat(label_path)['clas1']
    label_local = sio.loadmat(label_path_local)['clas1']
    row,col = label.shape
    get_random_sample(label,label_local)

def get_sample_npy(label_path):
    label = sio.loadmat(label_path)['clas1']
    with open('train.plk','rb') as op:
        lst = pickle.load(op)
    sample = 16*np.ones([1079,1024])
    cnt = 0
    for p in lst:
        cnt+=1
        sample[p[0],p[1]] = label[p[0],p[1]]-1
    print(cnt)
    np.save('sample.npy',sample)

def batch_yield(batch_size,data_path,is_shuffle,raw_real,raw_imag,label):
    with open(data_path,'rb') as op:
        data_lst = pickle.load(op)
    for i,tag in enumerate(label):
        label[i] = label[i]-1
    if is_shuffle:
        random.shuffle(data_lst)
    batch_num = len(data_lst)//batch_size
    print(batch_num)
    data_real,data_imag,label_batch = [],[],[]
    for i in range(batch_num):
        for p in data_lst[i*batch_size:(i+1)*batch_size]:
            tmp_r = image_scan(9,p,1079,1024,raw_real,True)
            tmp_i = image_scan(9,p,1079,1024,raw_imag,True)
            data_real.extend(tmp_r)
            data_imag.extend(tmp_i)
            label_batch.extend([label.item(p[0],p[1]) for i in range(3)])
        z = zip(data_real,data_imag,label_batch)
        del data_real[:]
        del data_imag[:]
        del label_batch[:]
        zz = zip(*z)
        del z[:]
        yield batch_num,zz[0],zz[1],zz[2]
        del zz[:]

def full_image_yield(image_path,label_path,window_size):
    data_real,data_imag = matReader(image_path)
    label = sio.loadmat(label_path)['clas1']
    row,col = label.shape
    for i in range(row):
        if i%10==0:
            print(i)
        point = [(i,j) for j in range(col)]
        yield  row,col,get_image_data(data_real,point,label_path,window_size,row,col),get_image_data(data_imag,point,label_path,window_size,row,col),label[i]


def get_image(row,col,label_pred,image_path,label_path="/home/iecas7/Project/polsar_segmentation/Flevoland/test1.mat"):
    image = Image.new("RGB",(row,col))
    image_local = Image.new("RGB",(row,col))
    label = sio.loadmat(label_path)['clas1']
    label = label-1
    label_local = np.zeros_like(label_pred)
    cnt = [0 for i in range(16)]
    for i in range(row):
        for j in range(col):
            cnt[label_pred[i][j]]+=1
            if label_pred[i][j]==0:
                image.putpixel([i,j],(0,132,40))
            elif label_pred[i][j]==1:
                image.putpixel([i,j],(41,118,167))
            elif label_pred[i][j]==2:
                image.putpixel([i,j],(149,122,101))
            elif label_pred[i][j]==3:
                image.putpixel([i,j],(244,234,41))
            elif label_pred[i][j]==4:
                image.putpixel([i,j],(255,153,102))
            elif label_pred[i][j]==5:
                image.putpixel([i,j],(185,52,40))
            elif label_pred[i][j]==6:
                image.putpixel([i,j],(102,153,51))
            elif label_pred[i][j]==7:
                image.putpixel([i,j],(115,179,207))
            elif label_pred[i][j]==8:
                image.putpixel([i,j],(255,153,255))
            elif label_pred[i][j]==9:
                image.putpixel([i,j],(174,206,114))
            elif label_pred[i][j]==10:
                image.putpixel([i,j],(134,45,109))
            elif label_pred[i][j]==11:
                image.putpixel([i,j],(214,150,60))
            elif label_pred[i][j]==12:
                image.putpixel([i,j],(255,215,179))
            elif label_pred[i][j]==13:
                image.putpixel([i,j],(0,164,255))
            elif label_pred[i][j]==14:
                image.putpixel([i,j],(204,204,204))
            elif label_pred[i][j]==15:
                image.putpixel([i,j],(255,51,51))
            if label[i][j]<16:
                label_local[i][j] = label_pred[i][j]
                color = image.getpixel((i,j))
                image_local.putpixel([i,j],color)
            else:
                label_local[i][j] = 16
    print(cnt)
    np.save("image_seg.npy",label_pred)
    np.save("image_seg_local.npy",label_local)
    image.save(image_path+'.jpg',quality=100)
    image_local.save(image_path+'_local.jpg',quality=100)


def show(path,label_path="/home/iecas7/Project/polsar_segmentation/Flevoland/test1.mat"):
    label_pred = np.load(path)
    row,col = label_pred.shape
    image = Image.new("RGB",(row,col))
    image_local = Image.new("RGB",(row,col))
    label = sio.loadmat(label_path)['clas1']
    label = label-1
    for i in range(row):
        for j in range(col):
            if label_pred[i][j]==0:
                image.putpixel([i,j],(0,132,40))
            elif label_pred[i][j]==1:
                image.putpixel([i,j],(41,118,167))
            elif label_pred[i][j]==2:
                image.putpixel([i,j],(149,122,101))
            elif label_pred[i][j]==3:
                image.putpixel([i,j],(244,234,41))
            elif label_pred[i][j]==4:
                image.putpixel([i,j],(255,153,102))
            elif label_pred[i][j]==5:
                image.putpixel([i,j],(185,52,40))
            elif label_pred[i][j]==6:
                image.putpixel([i,j],(102,153,51))
            elif label_pred[i][j]==7:
                image.putpixel([i,j],(115,179,207))
            elif label_pred[i][j]==8:
                image.putpixel([i,j],(255,153,255))
            elif label_pred[i][j]==9:
                image.putpixel([i,j],(174,206,114))
            elif label_pred[i][j]==10:
                image.putpixel([i,j],(134,45,109))
            elif label_pred[i][j]==11:
                image.putpixel([i,j],(214,150,60))
            elif label_pred[i][j]==12:
                image.putpixel([i,j],(255,215,179))
            elif label_pred[i][j]==13:
                image.putpixel([i,j],(0,164,255))
            elif label_pred[i][j]==14:
                image.putpixel([i,j],(204,204,204))
            elif label_pred[i][j]==15:
                image.putpixel([i,j],(255,51,51))
            if label[i][j]<16:
                color = image.getpixel((i,j))
                image_local.putpixel([i,j],color)
    image.show()

if __name__=="__main__":
    data_path = "/home/iecas7/Project/polsar_segmentation/Flevoland/filter/Flavoland"
    label_path = "/home/iecas7/Project/polsar_segmentation/Flevoland/label_local_full.mat"
    label_path_local = "/home/iecas7/Project/polsar_segmentation/Flevoland/test.mat"

    get_train_data(data_path,label_path,label_path_local)
    get_sample_npy(label_path)


    


