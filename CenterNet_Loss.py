import tensorflow as tf

from Define import *

def L1_Loss(vector_1, vector_2):
    return tf.abs(vector_1 - vector_2)

def Keypoints_Loss(pred_heatmaps, gt_heatmaps, gt_masks, alpha = 2.0, beta = 4.0):
    with tf.variable_scope('Keypoints'):
        pos_mask = gt_masks
        neg_mask = 1 - pos_mask

        pos_loss_op = -pos_mask * tf.pow(1 - pred_heatmaps, alpha) * tf.log(pred_heatmaps + 1e-10)
        neg_loss_op = -neg_mask * tf.pow(1 - gt_heatmaps, beta) * tf.pow(pred_heatmaps, alpha) * tf.log(1 - pred_heatmaps + 1e-10)

        keypoints_loss_op = tf.reduce_sum(pos_loss_op + neg_loss_op, axis = [1, 2, 3]) / tf.reduce_sum(pos_mask, axis = [1, 2, 3])
        keypoints_loss_op = tf.reduce_sum(keypoints_loss_op)

    return keypoints_loss_op

def Offsets_Loss(pred_offsets, gt_offsets, gt_masks):
    with tf.variable_scope('Offsets'):
        pos_mask = gt_masks
        offsets_loss_op = L1_Loss(pos_mask * pred_offsets, pos_mask * gt_offsets)

        offsets_loss_op = tf.reduce_sum(offsets_loss_op, axis = [1, 2, 3]) / tf.reduce_sum(pos_mask, axis = [1, 2, 3])
        offsets_loss_op = tf.reduce_sum(offsets_loss_op)

    return offsets_loss_op

def Sizes_Loss(pred_sizes, gt_sizes, gt_masks):
    with tf.variable_scope('Sizes'):
        pos_mask = gt_masks
        sizes_loss_op = L1_Loss(pos_mask * pred_sizes, pos_mask * gt_sizes)
        
        sizes_loss_op = tf.reduce_sum(sizes_loss_op, axis = [1, 2, 3]) / tf.reduce_sum(pos_mask, axis = [1, 2, 3])
        sizes_loss_op = tf.reduce_sum(sizes_loss_op)

    return sizes_loss_op

def CenterNet_Loss(pred_ops, gt_ops, gt_masks, lambda_offsets = 1, lambda_sizes = 0.1):
    pred_heatmaps, pred_offsets, pred_sizes = pred_ops
    gt_heatmaps, gt_offsets, gt_sizes = gt_ops

    kloss_op = Keypoints_Loss(pred_heatmaps, gt_heatmaps, gt_masks)
    oloss_op = lambda_offsets * Offsets_Loss(pred_offsets, gt_offsets, gt_masks)
    sloss_op = lambda_sizes * Sizes_Loss(pred_sizes, gt_sizes, gt_masks)
    
    loss_op = kloss_op + sloss_op + oloss_op
    return loss_op, kloss_op, oloss_op, sloss_op

if __name__ == '__main__':
    pred_heatmaps = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, CLASSES])
    pred_offsets = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2])
    pred_sizes = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2])

    gt_heatmaps = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, CLASSES])
    gt_offsets = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2])
    gt_sizes = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2])
    gt_masks = tf.placeholder(tf.float32, [None, OUTPUT_HEIGHT, OUTPUT_WIDTH, 1])

    pred_ops = [pred_heatmaps, pred_offsets, pred_sizes]
    gt_ops = [gt_heatmaps, gt_offsets, gt_sizes]

    loss_op, kloss_op, oloss_op, sloss_op = CenterNet_Loss(pred_ops, gt_ops, gt_masks)
    print(loss_op, kloss_op, oloss_op, sloss_op)

