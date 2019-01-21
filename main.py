import os
import tensorflow as tf
from CRPM_Net import CRPM_Net



#  network config

flags = tf.flags

flags.DEFINE_float("clip",                       80,          "Gradient clip")
flags.DEFINE_float("keep_prob",            0.5,         "Dropout rate")
flags.DEFINE_float("image_size",            22,         "image size")
flags.DEFINE_float("num_channel",        18,          "num_channel")
flags.DEFINE_float("num_tags",              16,          "tag_num")
flags.DEFINE_float("batch_size",             5,            "batch size") 
flags.DEFINE_float("test_batch_size",      10,         "batch size for test data")
flags.DEFINE_float("learning_rate",          0.01,        "Initial learning rate")
flags.DEFINE_float("regularation_rate",    0.001 ,        "Initial learning rate")
flags.DEFINE_float("decay_rate",             0.96,       "decay rate")
flags.DEFINE_string("model_path",          "model/model.ckpt",   "Path to save model")
flags.DEFINE_string("logger_path",          "train.log",    "File for log")
flags.DEFINE_string("optimizer",              "adam",     "Optimizer for training")
flags.DEFINE_integer("max_epoch",         1000,         "maximum training epochs")
flags.DEFINE_integer("steps_check",        100,        "steps per checkpoint")
flags.DEFINE_string("mode",                    "test2",     "mode of the LSTM Net")
flags.DEFINE_string("train_data_file",       "train.plk",  "Path for train data")
flags.DEFINE_string("dev_data_file",         "test.plk",    "Path for dev data")
flags.DEFINE_string("raw_path",               "data/Flavoland",      "Path for dev data")
flags.DEFINE_string("label_path",             "data/label.mat",      "Path for dev data")





FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 100.1, "gradient clip should't be too much"
assert 0 <= FLAGS.keep_prob < 1, "dropout rate between 0 and 1"
assert FLAGS.learning_rate > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]
assert FLAGS.mode in ["train", "test1","test2", "dilate"]
 
def main(_):
    #print("Please input the mode >> (train | test | demo)")
    #FLAGS.mode = raw_input()
    if FLAGS.mode == "train":
        model = CRPM_Net(batch_size=FLAGS.batch_size,image_size=FLAGS.image_size,raw_path=FLAGS.raw_path,
                          num_label=FLAGS.num_tags,regularation_rate=FLAGS.regularation_rate,learning_rate=FLAGS.learning_rate,
                          decay_rate=FLAGS.decay_rate,data_file=FLAGS.train_data_file,label_file=FLAGS.label_path,is_training=True,
                          dev_data_file=FLAGS.dev_data_file,clip=FLAGS.clip,num_epochs=FLAGS.max_epoch,logger_path=FLAGS.logger_path,model_path=FLAGS.model_path)
        model.build_graph()
        model.train()
    elif FLAGS.mode == "dilate":
        model = CRPM_Net(batch_size=FLAGS.batch_size,image_size=FLAGS.image_size,raw_path=FLAGS.raw_path,
                          num_label=FLAGS.num_tags,regularation_rate=FLAGS.regularation_rate,learning_rate=FLAGS.learning_rate,
                          decay_rate=FLAGS.decay_rate,data_file=FLAGS.train_data_file,label_file=FLAGS.label_path,is_training=False,
                          dev_data_file=FLAGS.dev_data_file,clip=FLAGS.clip,
                          num_epochs=FLAGS.max_epoch,logger_path=FLAGS.logger_path,model_path=FLAGS.model_path)
        model.get_dilated_inference()
        sess = model.model_restore()
        model.test_dilate(sess)
    elif FLAGS.mode == "test1":
        model = CRPM_Net(batch_size=FLAGS.batch_size,image_size=FLAGS.image_size,raw_path=FLAGS.raw_path,
                          num_label=FLAGS.num_tags,regularation_rate=FLAGS.regularation_rate,learning_rate=FLAGS.learning_rate,
                          decay_rate=FLAGS.decay_rate,data_file=FLAGS.train_data_file,label_file=FLAGS.label_path,is_training=False,
                          dev_data_file=FLAGS.dev_data_file,clip=FLAGS.clip,
                          num_epochs=FLAGS.max_epoch,logger_path=FLAGS.logger_path,model_path=FLAGS.model_path)
        model.build_graph()
        sess = model.model_restore()
        model.test_image(sess,FLAGS.image_path,FLAGS.image_label_path,9)
    elif FLAGS.mode == "test2":
        model = CRPM_Net(batch_size=FLAGS.batch_size,image_size=FLAGS.image_size,raw_path=FLAGS.raw_path,
                          num_label=FLAGS.num_tags,regularation_rate=FLAGS.regularation_rate,learning_rate=FLAGS.learning_rate,
                          decay_rate=FLAGS.decay_rate,data_file=FLAGS.train_data_file,label_file=FLAGS.label_path,is_training=False,
                          dev_data_file=FLAGS.dev_data_file,clip=FLAGS.clip,
                          num_epochs=FLAGS.max_epoch,logger_path=FLAGS.logger_path,model_path=FLAGS.model_path)
        model.build_graph()
        sess = model.model_restore()
        model.test(sess)

if __name__=="__main__":
    tf.app.run()
