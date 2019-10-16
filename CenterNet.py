import numpy as np
import tensorflow as tf

def Decode(pred_heatmaps, pred_offsets, pred_sizes, centers):
    mask = tf.layers.max_pooling2d(inputs = pred_heatmaps, pool_size = [3, 3], strides = 1, padding = 'same')
    mask = tf.cast(tf.equal(pred_heatmaps, mask), dtype = tf.float32)
    pred_heatmaps = mask * pred_heatmaps

    b, h, w, c = pred_heatmaps.shape.as_list()
    pred_centers = tf.reshape(centers + pred_offsets, [-1, w * h, 2])
    pred_sizes = tf.reshape(pred_sizes, [-1, w * h, 2])

    pred_lt = pred_centers - pred_sizes / 2
    pred_rb = pred_centers + pred_sizes / 2
    pred_bboxes = tf.concat([pred_lt, pred_rb], axis = -1)

    return pred_heatmaps, pred_bboxes
