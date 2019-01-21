import numpy as np 
from PIL import Image
import scipy.io as sio
import random
from eval_segm import*
from utils import *



def validation_yield(data_r,data_i,label,image_size=256):
    row,col = label.shape
    data_lst_r,data_lst_i,label_lst = [[],[]],[[],[]],[]
    ps = image_split1(row,col,image_size,image_size)
    for p in ps:
        if p[2]==0:
            data_lst_r[0].append(image_add_border(data_r[p[0]:p[0]+image_size,p[1]:p[1]+image_size,:],[image_size+6,image_size+6]))
            data_lst_r[1].append(data_r[p[0]:p[0]+image_size,p[1]:p[1]+image_size,:])
            data_lst_i[0].append(image_add_border(data_i[p[0]:p[0]+image_size,p[1]:p[1]+image_size,:],[image_size+6,image_size+6]))
            data_lst_i[1].append(data_i[p[0]:p[0]+image_size,p[1]:p[1]+image_size,:])
            label_lst.append(label[p[0]:p[0]+image_size,p[1]:p[1]+image_size])
    return data_lst_r,data_lst_i,label_lst


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


def image_test(image_path,label_path):
    im_r,im_i = matReader(image_path)
    label = sio.loadmat(label_path)['clas1']
    label-=1
    im_array_r = np.asarray(im_r)
    im_array_i = np.asarray(im_i)
    image_list_r,image_list_i = [],[]
    label_list = []
    (row,col,channel) = np.shape(im_array_r)
    image_list_r.append(im_array_r[:1076,:,:])
    image_list_i.append(im_array_i[:1076,:,:])
    label_list.append(label)
    return np.asarray(image_list_r),np.asarray(image_list_i),np.asarray(label_list)

def image_save(label_pred,row,col,image_path,label_path,is_total=False,):
    image = Image.new("RGB",(col,row))
    image_local = Image.new("RGB",(col,row))
    label = sio.loadmat(label_path)['clas1']
    label_local = np.zeros_like(label_pred)
    label = label-1
    for i in range(row):
        for j in range(col):
            if label_pred[i][j]==0:
                image.putpixel([j,i],(0,132,40))
            elif label_pred[i][j]==1:
                image.putpixel([j,i],(41,118,167))
            elif label_pred[i][j]==2:
                image.putpixel([j,i],(149,122,101))
            elif label_pred[i][j]==3:
                image.putpixel([j,i],(244,234,41))
            elif label_pred[i][j]==4:
                image.putpixel([j,i],(255,153,102))
            elif label_pred[i][j]==5:
                image.putpixel([j,i],(185,52,40))
            elif label_pred[i][j]==6:
                image.putpixel([j,i],(102,153,51))
            elif label_pred[i][j]==7:
                image.putpixel([j,i],(115,179,207))
            elif label_pred[i][j]==8:
                image.putpixel([j,i],(255,153,255))
            elif label_pred[i][j]==9:
                image.putpixel([j,i],(174,206,114))
            elif label_pred[i][j]==10:
                image.putpixel([j,i],(134,45,109))
            elif label_pred[i][j]==11:
                image.putpixel([j,i],(214,150,60))
            elif label_pred[i][j]==12:
                image.putpixel([j,i],(255,215,179))
            elif label_pred[i][j]==13:
                image.putpixel([j,i],(0,164,255))
            elif label_pred[i][j]==14:
                image.putpixel([j,i],(204,204,204))
            elif label_pred[i][j]==15:
                image.putpixel([j,i],(255,51,51))
            if is_total:
                if label[i][j]<16:
                    label_local[i][j] = label_pred[i][j]
                    color = image.getpixel((j,i))
                    image_local.putpixel([j,i],color)
                else:
                    label_local[i][j] = 16
    if is_total:
        image_local.save(image_path+'_local.jpg',quality=100)
        image.save(image_path+'.jpg',quality = 100)
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        np.save(image_path+'.npy',label_pred)
        np.save(image_path+'_local.npy',label_local)
        return image_path+'_local.npy'
    else:
        image.save(image_path,quality = 100)


def get_sample(data_r,data_i,label,points,image_size):
    data_lst,data_lst_raw,label_lst = [[],[]],[[],[]],[]
    for p in points:
        x_raw_r = data_r[p[0]:p[0]+image_size,p[1]:p[1]+image_size,:]
        x_raw_i = data_i[p[0]:p[0]+image_size,p[1]:p[1]+image_size,:]
        x_r = image_add_border(x_raw_r,[image_size+6,image_size+6])
        x_i = image_add_border(x_raw_i,[image_size+6,image_size+6])
        l = label[p[0]:p[0]+image_size,p[1]:p[1]+image_size]
        if p[2]==0:
            data_lst[0].append(x_r)
            data_lst[1].append(x_i)
            data_lst_raw[0].append(x_raw_r)
            data_lst_raw[1].append(x_raw_i)
            label_lst.append(l)
        elif p[2]==1:
            a,b = rot(np.copy(x_raw_r),np.copy(l),1)
            data_lst_raw[0].append(a)
            data_lst[0].append(image_add_border(a,[image_size+6,image_size+6]))
            label_lst.append(b)
            a,b = rot(np.copy(x_raw_i),np.copy(l),1)
            data_lst_raw[1].append(a)
            data_lst[1].append(image_add_border(a,[image_size+6,image_size+6]))
        else:
            a,b = flip(np.copy(x_raw_r),np.copy(l))
            data_lst_raw[0].append(a)
            data_lst[0].append(image_add_border(a,[image_size+6,image_size+6]))
            label_lst.append(b)
            a,b = flip(np.copy(x_raw_i),np.copy(l))
            data_lst_raw[1].append(a)
            data_lst[1].append(image_add_border(a,[image_size+6,image_size+6]))
    return data_lst,data_lst_raw,label_lst


def image_add_border(img,image_size):
    b_size = [(image_size[0]-img.shape[0])/2,(image_size[1]-img.shape[1])/2]
    img_new = np.zeros([image_size[0],image_size[1],img.shape[2]])
    img_new[b_size[0]:image_size[0]-b_size[0],b_size[1]:image_size[1]-b_size[1],:] = img
    img_new[:b_size[0],b_size[1]:image_size[1]-b_size[1],:] = np.flipud(img[:b_size[0],:,:])
    img_new[image_size[0]-b_size[0]:,b_size[1]:image_size[1]-b_size[1],:] = np.flipud(img[img.shape[0]-b_size[0]:,:,:])
    img_new[:,:b_size[1],:] = np.fliplr(img_new[:,b_size[1]:b_size[1]*2,:])
    img_new[:,image_size[1]-b_size[1]:,:] = np.fliplr(img_new[:,image_size[1]-2*b_size[1]:image_size[1]-b_size[1],:])
    return img_new


def image_yield(raw_r,raw_i,label,stride=128,image_size=516,batch_size=1,pyramid=False):
    row,col = label.shape
    image_list = image_split1(row,col,stride,image_size)
    print("data shuffling...")
    random.shuffle(image_list)
    print("data shuffling complete")
    #label_list = [label_list[i] for i in shuffle_list]
    num_batch = len(image_list)/batch_size
    for ind in range(num_batch):
        batch_list,batch_list_raw,batch_label_lst = get_sample(raw_r,raw_i,label,image_list[ind*batch_size:(ind+1)*batch_size],image_size)
        batch_img_r = [np.asarray(batch_list[0]),np.asarray(batch_list_raw[0])]
        batch_img_i = [np.asarray(batch_list[1]),np.asarray(batch_list_raw[1])]
        batch_label = np.asarray(batch_label_lst)
        yield num_batch,batch_img_r,batch_img_i,batch_label










