import tensorflow as tf
import numpy as np



def loss_dice(logits, labels, num_classes,batch_size_tf):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height].
          The ground truth of your data.
      weights: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    #labels=tf.squeeze(labels)
    with tf.name_scope('loss'):
        #shapelables=labels.get_shape()