import os
import tensorflow as tf
import pprint
from g_model import seg_GAN
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("adverarial", False, "Adversarial or normal [50000]")
flags.DEFINE_integer("iterations", 500000, "