import tensorflow as tf
import numpy as np



def loss_dice(logits, labels, num_classes,batch_size_tf):
    """Calculate the loss from th