import tensorflow as tf 
import datetime
from tensorflow.python import debug as tf_debug
from utils import batch_yield,get_logger,full_image_yield,get_image
from image_split import image_yield,image_save,image_test,validation_yield
from net_utils import complex_cross_dilated_conv,crop_and_concat,max_pool,focal_loss,complex_cross_conv,complex_cross_deconv,complex_cross_fc
from collections import OrderedDict
from utils import *
from math import log
import os
from evaluator import evaluator

class CRPM_Net(object):
    def __init__(self,batch_size,image_size,raw_path,num_label,regularation_rate,learning_rate,logger_path,
                 decay_rate,is_training,data_file,label_file,dev_data_file,clip,num_epochs,model_path):
        '''
            standard:   training standard. 1: Cs-CNN 2: CRPM-Net
            channel:   Input data channel. 18: L,P,C-band complex_cross-value [C]. 27: L,P,C-band real-value [C].
            filter_size: Base convlution kernel size
            features_root: the channel number of first convolution layer
            image_size: Sliding window size for CRPM-Net
            stride: Slideing wondow size for CRPM-Net
        '''
        self.data_path = data_file
        self.label_path = label_file
        self.dev_data_path = dev_data_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_label = num_label
        self.model_path = model_path
        self.clip = clip
        self.standard = 2  
        self.layers = 3
        self.filter_size = 3
        self.channels = 18 
        self.features_root = 12
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.is_training = is_training 
        self.stride=64
        self.raw_path=raw_path
        self.image_size=128
        self.logger = get_logger(logger_path)
        self.regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    def build_graph(self):
        self.add_placeholder()
        if self.standard == 1:
            self.inference_Cs_CNN()
        else:
            self.inference_CRPM_Net()
        
        if self.standard == 1:
            self.loss_focal_loss()
            self.train_step_Cs_CNN()
        else:
            self.loss()
            self.train_step_CRPM_Net()

    def add_placeholder(self):
        self.input_real = tf.placeholder(tf.float32,shape=[None,None,None,self.channels],name="batch_real")
        self.input_imag = tf.placeholder(tf.float32,shape=[None,None,None,self.channels],name="batch_imag")
        self.decay_steps = tf.placeholder(tf.int32,name="decay_steps")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout for Cs-CNN
        if self.standard==2:
            self.keep_prob2 = tf.placeholder(tf.float32, name="dropout2_probability")  # dropout for CRPM-Net

    def down_conv(self):
        #Cs-CNN inference
        dw_h_convs = OrderedDict()
        pools = OrderedDict()
        x = [self.input_real,self.input_imag]
        for layer in range(self.layers):
            features = 2 ** layer * self.features_root
            if layer == 0:
                conv1_real,conv1_imag = complex_cross_conv(input_real=x[0],input_imag=x[1],
                                                                            scope_name="down_conv_"+str(layer)+'/'+'conv1',
                                                                            input_shape=[self.filter_size, self.filter_size, self.channels, features],
                                                                            keep_prob=self.keep_prob,padding='VALID',regularizer=self.regularizer)
                print([self.filter_size, self.filter_size, self.channels, features])
            elif layer<self.layers-1:
                conv1_real,conv1_imag = complex_cross_conv(input_real=x[0],input_imag=x[1],
                                                                            scope_name="down_conv_"+str(layer)+'/'+'conv1',
                                                                            input_shape=[self.filter_size, self.filter_size, features // 2, features],
                                                                            keep_prob=self.keep_prob,padding='VALID',regularizer=self.regularizer)
                print([self.filter_size, self.filter_size, features // 2, features])
            else:
                conv1_real,conv1_imag = complex_cross_conv(input_real=x[0],input_imag=x[1],
                                                                            scope_name="down_conv_"+str(layer)+'/'+'conv1',
                                                                            input_shape=[1,1, features // 2, features],
                                                                            keep_prob=self.keep_prob,padding='VALID',regularizer=self.regularizer)
            dw_h_convs[layer]  = [conv1_real,conv1_imag]
            if layer < self.layers - 1:
                pools[layer] = [max_pool(dw_h_convs[layer][0], 2),max_pool(dw_h_convs[layer][1], 2)]
                x = pools[layer]
        return dw_h_convs

    def down_atrous_conv(self):
        #C-Dilated-CNN inference
        self.input_raw_r = tf.placeholder(tf.float32,shape=[None,None,None,self.channels],name="batch_images_uncroped_r")
        self.input_raw_i = tf.placeholder(tf.float32,shape=[None,None,None,self.channels],name="batch_images_uncroped_i")
        dw_h_convs = OrderedDict()
        pools = OrderedDict()
        x = [self.input_raw_r,self.input_raw_i]
        for layer in range(self.layers):
            print(layer)
            print(x[0].get_shape().as_list())
            features = 2 ** layer * self.features_root
            if layer == 0:
                conv1_real,conv1_imag = complex_cross_conv(input_real=x[0],input_imag=x[1],
                                                                            scope_name="down_conv_"+str(layer)+'/'+'conv1',
                                                                            input_shape=[self.filter_size, self.filter_size, self.channels, features],
                                                                            keep_prob=self.keep_prob,padding='SAME',regularizer=self.regularizer)
                print([self.filter_size, self.filter_size, self.channels, features])
            elif layer<self.layers-1:
                conv1_real,conv1_imag = complex_cross_dilated_conv(input_real=x[0],input_imag=x[1],
                                                                            scope_name="down_conv_"+str(layer)+'/'+'conv1',
                                                                            input_shape=[self.filter_size, self.filter_size, features // 2, features],
                                                                            keep_prob=self.keep_prob,regularizer=self.regularizer)
                print([self.filter_size, self.filter_size, features // 2, features])
            else:
                conv1_real,conv1_imag = complex_cross_conv(input_real=x[0],input_imag=x[1],
                                                                            scope_name="down_conv_"+str(layer)+'/'+'conv1',
                                                                            input_shape=[1,1, features // 2, features],
                                                                            keep_prob=self.keep_prob,padding='SAME',regularizer=self.regularizer)

            dw_h_convs[layer]  = [conv1_real,conv1_imag]
            print([self.filter_size, self.filter_size, features, features])
            print('\n')
            if layer < self.layers - 1:
                pools[layer] = [max_pool(dw_h_convs[layer][0], 2,1,"SAME"),max_pool(dw_h_convs[layer][1], 2,1,'SAME')]
                x = pools[layer]
        return dw_h_convs


    def inference_Cs_CNN(self):
        self.label = tf.placeholder(tf.int32,shape=[None],name="image_label")
        self.up_h_convs = OrderedDict()
        dw_h_convs = self.down_conv()
        x = dw_h_convs[self.layers - 1]       
        flat_real = tf.reshape(x[0],[tf.shape(self.input_real)[0],-1])
        flat_imag = tf.reshape(x[1],[tf.shape(self.input_real)[0],-1])
        #input_real,input_imag,scope_name,input_shape,isActive
        fc_real,fc_imag = complex_cross_fc(flat_real,flat_imag,'fc',[self.features_root*4,self.num_label],False)
        logits_real = tf.expand_dims(fc_real,-1)
        logits_imag = tf.expand_dims(fc_imag,-1)
        self.real = logits_real
        self.imag = logits_imag
        self.mold = tf.sqrt(tf.add(tf.square(logits_real),tf.square(logits_imag)))
        self.phase = tf.atan(tf.div(logits_imag,tf.add(logits_real,tf.constant(1e-8))))
        self.logits = tf.concat((logits_real,logits_imag,self.mold,self.phase),-1)
        #[batch_size,num_label]
        with tf.variable_scope("mold_phase"):
            self.weight = tf.get_variable(
                    name="weight_model",
                    shape=[4,1],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
            self.logits = tf.reshape(tf.matmul(tf.reshape(self.logits,[-1,4]),self.weight),[tf.shape(self.input_real)[0],self.num_label])
        self.out = tf.argmax(self.logits,1)

    def inference_CRPM_Net(self):
        self.label = tf.placeholder(tf.int32,shape=[None,None,None],name="image_label")
        self.up_h_convs = OrderedDict()
        deconv_dic = OrderedDict()

        # Encoder network   Cs-CNN
        dw_h_convs = self.down_conv()
        x = dw_h_convs[self.layers - 1]
        flat_real = tf.reshape(x[0],[-1,self.features_root*4])
        flat_imag = tf.reshape(x[1],[-1,self.features_root*4])
        fc_real,fc_imag = complex_cross_fc(flat_real,flat_imag,'fc',[self.features_root*4,self.num_label],False)
        heat_x = [tf.reshape(fc_real,[tf.shape(x[0])[0],tf.shape(x[0])[1],tf.shape(x[0])[2],self.num_label]),tf.reshape(fc_imag,[tf.shape(x[1])[0],tf.shape(x[1])[1],tf.shape(x[1])[2],self.num_label])]
        #C-Dilated CNN
        dw_h_convs_atrous = self.down_atrous_conv()

        if self.is_training:
            x_atrous = dw_h_convs_atrous[self.layers-1]
            flat_real_atrous = tf.reshape(x_atrous[0],[-1,self.features_root*4])
            flat_imag_atrous = tf.reshape(x_atrous[1],[-1,self.features_root*4])    

            fc_real_atrous,fc_imag_atrous = complex_cross_fc(flat_real_atrous,flat_imag_atrous,'fc',[self.features_root*4,self.num_label],False)
            
            logits_real_atrous = tf.expand_dims(fc_real_atrous,-1)
            logits_imag_atrous = tf.expand_dims(fc_imag_atrous,-1)
            mold_atrous = tf.sqrt(tf.add(tf.square(logits_real_atrous),tf.square(logits_imag_atrous)))
            phase_atrous = tf.atan(tf.div(logits_imag_atrous,tf.add(logits_real_atrous,tf.constant(1e-8))))
            self.logits_atrous = tf.concat((logits_real_atrous,logits_imag_atrous,mold_atrous,phase_atrous),-1)
            
            with tf.variable_scope("mold_phase"):
                weight_atrous1 = tf.get_variable(
                        name="weight_model",
                        shape=[4,1],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32)

            # Fusion of score map of C-Dilated CNN and training pixels
            self.logits_atrous = tf.reshape(tf.matmul(tf.reshape(self.logits_atrous,[-1,4]),weight_atrous1),[tf.shape(x_atrous[0])[0],tf.shape(x_atrous[0])[1],tf.shape(x_atrous[0])[2],self.num_label])            
            self.label_dilate = tf.cast(tf.argmax(self.logits_atrous,3),tf.int32)
            b_ =  tf.scalar_mul(16, tf.ones_like(self.label, dtype=tf.int32))
            self.weight_loss = tf.where(tf.equal(self.label,b_),tf.scalar_mul(0,tf.ones_like(self.label, dtype=tf.float32)),
                                    tf.scalar_mul(1, tf.ones_like(self.label, dtype=tf.float32)))
            self.label_new = tf.where(tf.equal(self.label,b_),self.label_dilate,self.label)
            self.weight_loss = tf.where(tf.not_equal(self.label_new,self.label_dilate), tf.scalar_mul(1, tf.ones_like(self.label, dtype=tf.float32)),
                                    self.weight_loss)

        # Decoder network
        for layer in range(self.layers-2,-1,-1):
            features = 2 ** (layer + 1) * self.features_root
            if layer==self.layers-2:
                h_deconv_r,h_deconv_i = complex_cross_deconv(input_real=heat_x[0],input_imag=heat_x[1],
                                                                scope_name="up_conv_"+str(layer)+'/',
                                                                input_shape=[2, 2, features // 2, self.num_label])
            else:
                h_deconv_r,h_deconv_i = complex_cross_deconv(input_real=heat_x[0],input_imag=heat_x[1],
                                                                scope_name="up_conv_"+str(layer)+'/',
                                                                input_shape=[2, 2, features // 2, features])
            h_deconv_concat = [crop_and_concat(dw_h_convs[layer][0], h_deconv_r),crop_and_concat(dw_h_convs[layer][1], h_deconv_i)]
            deconv_dic[layer] = h_deconv_concat
            if layer==0:
                x_dilate = dw_h_convs_atrous[self.layers-2]

                h_deconv_concat = [crop_and_concat(h_deconv_concat[0], x_dilate[0]),crop_and_concat(h_deconv_concat[1], x_dilate[1])]
                conv1_r,conv1_i = complex_cross_conv(input_real=h_deconv_concat[0],input_imag=h_deconv_concat[1],
                                                scope_name="up_conv_"+str(layer)+'/'+'conv1',
                                                input_shape=[self.filter_size, self.filter_size, features*2, features // 2],
                                                keep_prob=self.keep_prob2,padding='SAME',regularizer=self.regularizer)
            else:
                conv1_r,conv1_i = complex_cross_conv(input_real=h_deconv_concat[0],input_imag=h_deconv_concat[1],
                                                scope_name="up_conv_"+str(layer)+'/'+'conv1',
                                                input_shape=[self.filter_size, self.filter_size, features, features // 2],
                                                keep_prob=self.keep_prob2,padding='SAME',regularizer=self.regularizer)

            heat_x[0],heat_x[1] = conv1_r,conv1_i;
            self.up_h_convs[layer] = heat_x
        
        conv_out_r,conv_out_i = complex_cross_conv(input_real=heat_x[0],input_imag=heat_x[1],
                                            scope_name='output_map/conv1x1',
                                            input_shape=[1, 1, self.features_root, self.num_label],
                                            keep_prob=tf.constant(1.0),padding='VALID',regularizer=self.regularizer)

        logits_real = tf.expand_dims(conv_out_r,-1)
        logits_imag = tf.expand_dims(conv_out_i,-1)
        self.mold = tf.sqrt(tf.add(tf.square(logits_real),tf.square(logits_real)))
        self.phase = tf.atan(tf.div(logits_imag,tf.add(logits_real,tf.constant(1e-8))))
        self.logits = tf.concat((logits_real,logits_imag,self.mold,self.phase),-1)
        #[batch_size,num_label]
        with tf.variable_scope("mold_phase_U_Net"):
            self.weight = tf.get_variable(
                    name="weight_model",
                    shape=[4,1],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
        self.logits = tf.reshape(tf.matmul(tf.reshape(self.logits,[-1,4]),self.weight),[tf.shape(self.input_real)[0],tf.shape(conv_out_r)[1],tf.shape(conv_out_r)[2],self.num_label])
        self.out = tf.argmax(self.logits,3)

    def loss_focal_loss(self):
        labels_one_hot = tf.one_hot(self.label,16)
        with tf.variable_scope("loss"):
            self.loss = focal_loss(prediction_tensor=self.logits,target_tensor=labels_one_hot)
            self.loss = tf.reduce_mean(self.loss)+tf.add_n(tf.get_collection('losses'))

    def loss(self):
        with tf.variable_scope("loss"):
            annotation = tf.expand_dims(self.label,-1,name="annotation")
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                    labels=self.label)
            self.loss = tf.reduce_mean(self.loss)+tf.add_n(tf.get_collection('losses'))

    def train_step_CRPM_Net(self):
        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        var_up = tf.trainable_variables()[17:]
        print(len(var_up))
        variables_names = [v.name for v in tf.trainable_variables()[17:]]
        print(variables_names)
        optim = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate,staircase=False))
        grads_and_vars = optim.compute_gradients(self.loss,var_up)
        grads_and_vars_clip = []
        for g,v in grads_and_vars:
            print(v.name)
            print(g)
            grads_and_vars_clip.append([tf.clip_by_value(g, -self.clip, self.clip), v])
        
        capped_grads_and_vars = []
        for g_v in grads_and_vars_clip:
            for var in var_up:
                if g_v[1]==var:
                    print(g_v[1].name)
                    capped_grads_and_vars.append((g_v[0],g_v[1]))

        self.train_op = optim.apply_gradients(capped_grads_and_vars, global_step=self.global_step)
    

    def train_step_Cs_CNN(self):
        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate,staircase=False))
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip, self.clip), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
 

    def restore_step2(self,sess):
        var_up = tf.trainable_variables()[:17]

        saver = tf.train.Saver(var_up)
        saver.restore(sess,ckpt_file)
        print('step1 model restored...')


    def model_restore(self):
        #model_saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        variables_names = [v.name for v in tf.trainable_variables()]
        print(variables_names)
        saver.restore(sess,self.model_path)
        print('model restored...')

        #model_saver.save(sess,self.model_path,global_step=0)
        
        return sess

    def test(self,sess):
        start_time = datetime.datetime.now()
        img_r,img_i,lab = image_test(self.raw_path,self.label_path)
        img_r_raw = image_add_border(img_r[0],[img_r[0].shape[0]+6,img_r[0].shape[1]+6])
        img_i_raw = image_add_border(img_i[0],[img_i[0].shape[0]+6,img_i[0].shape[1]+6])
        feed_dict={self.input_real:[img_r_raw],self.input_imag:[img_i_raw],self.input_raw_r:img_r,self.input_raw_i:img_i,self.label:lab,self.decay_steps:1,self.keep_prob:1.,self.keep_prob2:1.}
        
        pred_label = sess.run(self.out, feed_dict=feed_dict)
        print("image_size: "+str(pred_label.shape[1:]))
        print("classification time: "+(str(datetime.datetime.now()-start_time)[5:])+' (s)')
        print('classification image saving...')
        pred_file = image_save(pred_label[0,...],pred_label.shape[1],pred_label.shape[2],"CRPM_Net",self.label_path,True)
        evaluator(pred_file,self.label_path)


    def test_dilate(self,sess):
        img_r,img_i,lab = image_test(self.raw_path,self.label_path)
        feed_dict={self.input_raw_r:img_r,self.input_raw_i:img_i,self.decay_steps:1,self.keep_prob:1.,self.keep_prob2:1.}
        start_time = datetime.datetime.now()
        [pred_label] = sess.run([self.label_dilate], feed_dict=feed_dict)
        print(pred_label.shape)
        image_save(pred_label[0,...],pred_label.shape[1],pred_label.shape[2],"dilated_pred",self.label_path,True)
        print("Evaluating...")
        print("%d s costed"%((datetime.datetime.now()-start_time).seconds))

    def train(self):
        model_saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess: 
            loss_lst,acc_lst,acc_cls_lst = [],[],[]
            sess.run(tf.global_variables_initializer())
            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            for k,v in zip(variables_names,values):
                print("Variable: ",k)
                print("Shape: ",v.shape)
            print(variables_names[:17])
            if self.standard == 2:
                self.restore_step2(sess)
            for epoch in range(self.num_epochs):
                global_step = sess.run(self.global_step)
                loss,acc = self.run_one_epoch(sess,model_saver,epoch+1,global_step,0.75)#keep_prob
                loss_lst.append(loss)
                acc_lst.append(acc)
                if epoch and epoch%100==0:
                    np.save('loss.npy',np.array(loss_lst))
                    np.save('acc.npy',np.array(acc_lst))


    def run_one_epoch(self,sess,model_saver,epoch,global_step,keep_prob):
        loss_total = []
        accuracy = 0
        print("global_step: %d"%global_step)
        if epoch==1:
            is_shuffle = False
        else:
            is_shuffle = True
        print('****************epoch start****************')
        data_real,data_imag = matReader(self.raw_path)
        label = sio.loadmat(self.label_path)['clas1']
        label_sample = np.load('sample.npy')
        if self.standard==2:
            batch_generator = image_yield(data_real,data_imag,label_sample,self.stride,self.image_size,self.batch_size,False)
        else:
            batch_generator = batch_yield(self.batch_size,self.data_path,is_shuffle,data_real,data_imag,label)
        #batch_generator = batch_yield(self.batch_size,self.data_path,is_shuffle)
        start_time = datetime.datetime.now()
        for step,(batch_num,image_batch_real,image_batch_imag,label_batch) in enumerate(batch_generator,1):
            if self.standard==2:
                feed_dict={self.input_real:image_batch_real[0],self.input_raw_r:image_batch_real[1],
                                    self.input_imag:image_batch_imag[0],self.input_raw_i:image_batch_imag[1],
                                    self.label:label_batch,self.decay_steps:batch_num,self.keep_prob:keep_prob,
                                    self.keep_prob2:keep_prob}
            else:
                feed_dict={self.input_real:image_batch_real,self.input_imag:image_batch_imag,self.label:label_batch,self.decay_steps:batch_num,self.keep_prob:keep_prob}
            
            _,loss,pred_label,steps = sess.run([self.train_op,self.loss,self.out,self.global_step],feed_dict=feed_dict)
            loss_total.append(loss)
            if step%10==0:
                print("%d / %d batch(s) processed in %dth epoch used %d sec and loss is %.4f" %(step,batch_num,epoch,(datetime.datetime.now()-start_time).seconds,loss))
                start_time = datetime.datetime.now()
               
            if step==batch_num:
                self.logger.info("%s >> %d epoch(s), %d step(s)" %(datetime.datetime.now(),epoch,step))
                if self.standard==2:
                    if epoch%5==0:
                        d_r,d_i,l= validation_yield1(data_real,data_imag,label_sample,self.image_size)
                        if self.standard==2:
                            feed_dict={self.input_real:d_r[0],self.input_raw_r:d_r[1],
                                                self.input_imag:d_i[0],self.input_raw_i:d_i[1],
                                                self.label:l,self.decay_steps:batch_num,
                                                self.keep_prob:1.,self.keep_prob2:1.}
                        else:
                            feed_dict={self.input_real:d_r,self.input_imag:d_i,self.label:l,self.decay_steps:batch_num,self.keep_prob:1.}
                        loss1,pred_label,label_1 = sess.run([self.loss,self.out,self.label_new],feed_dict=feed_dict)
                        correct_predict = tf.equal(pred_label,label_1)
                        accuracy1 = sess.run(tf.reduce_mean(tf.cast(correct_predict,tf.float32)))
                        
                        self.logger.info("**********train batch evaluation**********")
                        self.logger.info(">> %dth train_batch in %d epoch(s) / Accuracy:  %.4f / Loss: %.4f" %(step,epoch,accuracy1,loss1))
                        for i in range(len(l)):
                            image_save(pred_label[i,...],l[i].shape[0],l[i].shape[0],"sample_out/image1"+str(i)+"_pred.jpg")
                            image_save(label_1[i,...],label_1[i].shape[0],label_1[i].shape[0],"sample_out/image1"+str(i)+"_corr.jpg")
                        #cv2.imwrite("image_1"+str(i)+"_real.jpg",image_batch[i,...])
                elif epoch%1==0:
            
                    weight = sess.run([self.weight])
                    print(weight)
                    label_corr,loss,label_cls = self.test_pred(sess,100,self.dev_data_path)
                    accuracy = label_corr[0]/float(label_corr[1])
                    acc_cls = []
                    tag = ['grass: ','flax: ','potato: ','wheat: ','rapessed: ','beet: ','barley: ','peas: ','maize: ','bean: ','fruit: ','onion: ','oat: ','lucerne: ','building: ','road: ']
                    for i in range(16):
                        acc_cls.append(label_cls[i][0]/float(label_cls[i][1]))
                        print(tag[i]+"%4f"%(acc_cls[i]))
                    self.logger.info("**********validation evaluation**********")
                    self.logger.info("Validation >> %dth step in %d epoch(s) / Accuracy:  %.4f / Loss: %.4f" %(step,epoch,accuracy,loss))
            if step==batch_num and epoch%5==0:
                model_saver.save(sess, self.model_path,global_step=epoch)
        return np.mean(np.array(loss_total)),accuracy
    


   



                


