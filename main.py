import os
import tensorflow as tf
import pprint
from g_model import seg_GAN
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("adverarial", False, "Adversarial or normal [50000]")
flags.DEFINE_integer("iterations", 500000, "Epoch to train [50000]")
flags.DEFINE_float("learning_rate", 1e-8, "Learning rate of for SGD [1e-8]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("show_every", 10, "The size of batch images [10]")
flags.DEFINE_integer("save_every", 1000, "save every [1000]")
flags.DEFI