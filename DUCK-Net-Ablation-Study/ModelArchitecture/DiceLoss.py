import tensorflow as tf

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = tf.cast(ground_truth, tf.float32)
    predictions = tf.cast(predictions, tf.float32)
    
    ground_truth = tf.reshape(ground_truth, [-1])  # Flatten
    predictions = tf.reshape(predictions, [-1])    # Flatten

    intersection = tf.reduce_sum(predictions * ground_truth)
    union = tf.reduce_sum(predictions) + tf.reduce_sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice
