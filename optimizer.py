import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Optimizer(object):
    def __init__(self, preds, labels, preds2, labels2, Z1, ZC1, Z2, ZC2, Coef, SZ1, SZ2, PL, Theta, Labels, pos_weight1,
                 pos_weight2):
        preds_sub = preds
        labels_sub = labels
        preds_sub2 = preds2
        labels_sub2 = labels2

        # Reconstruction Loss
        self.cost1 = 0.5 * tf.reduce_sum(
            tf.nn.weighted_cross_entropy_with_logits(logits=tf.reshape(preds_sub, [-1]),
                                                     targets=tf.reshape(labels_sub, [-1]), pos_weight=pos_weight1))
        self.cost2 = 0.5 * tf.reduce_sum(
            tf.nn.weighted_cross_entropy_with_logits(logits=tf.reshape(preds_sub2, [-1]),
                                                     targets=tf.reshape(labels_sub2, [-1]), pos_weight=pos_weight2))
        self.cost = self.cost1 + self.cost2
        # Self-Expression Loss
        self.SEloss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(Z1, ZC1), 2)) + 0.5 * tf.reduce_sum(
            tf.pow(tf.subtract(Z2, ZC2), 2))
        self.S_Regular = tf.reduce_sum(tf.pow(tf.abs(Coef), 1.0))
        # Cluster-specificity Distribution
        # self.csd1 = tf.norm(tf.norm(Coef, ord=1, axis=0, keep_dims=True), ord=2, keep_dims=False)
        self.csd2 = tf.norm(tf.norm(tf.transpose(Coef), ord=1, axis=0, keep_dims=True), ord=2, keep_dims=False)
        self.csd = self.csd2 # self.csd1 +
        # Consist loss
        self.consistent_loss = tf.reduce_sum(tf.pow(tf.subtract(Z1, Z2), 2.0))
        # Pre-Train Total loss
        self.loss = 1e-3 * self.cost + 300 * self.SEloss + 0.1 * self.consistent_loss + 100 * self.S_Regular
        self.Pre_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.Pre_opt_op = self.Pre_optimizer.minimize(self.loss)
        self.grads_vars = self.Pre_optimizer.compute_gradients(self.loss)
        # Self-supervised loss
        self.cross_entropy1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=SZ1, labels=PL))
        self.cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=SZ2, labels=PL))
        self.dense_loss = self.cross_entropy1 + self.cross_entropy2
        # Cq Loss
        self.Cq_loss = tf.reduce_sum(tf.pow(tf.abs(tf.transpose(Coef) * Theta), 1.0))
        # Center Loss
        center_loss1, centers1, centers_update_op1 = self.get_center_loss1(features=SZ1, labels=Labels,
                                                                           alpha=0.5, num_classes=3)
        center_loss2, centers2, centers_update_op2 = self.get_center_loss2(features=SZ2, labels=Labels,
                                                                           alpha=0.5, num_classes=3)
        self.center_loss1 = tf.reduce_sum(center_loss1)
        self.center_loss2 = tf.reduce_sum(center_loss2)
        self.center_loss = self.center_loss1 + self.center_loss2
        # lambda_1 lambda_2 lambda_5 lambda_4
        self.Total_Loss = 1e-3 * self.cost + 10 * self.SEloss + 1e-4 * self.consistent_loss \
                          + 8 * self.Cq_loss + 10 * (self.dense_loss + 0.8 * self.center_loss) + 10*self.csd
        with tf.control_dependencies([centers_update_op1, centers_update_op2]):
            self.Fin_optimizer = tf.train.AdamOptimizer(learning_rate=3e-5)
            self.Fin_opt_op = self.Fin_optimizer.minimize(self.Total_Loss)


    def get_center_loss1(self, features, labels, alpha, num_classes):
        len_features = features.get_shape()[1]
        centers = tf.get_variable('centers1', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)
        diff = centers_batch - features
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        centers_update_op = tf.scatter_sub(centers, labels, diff)
        return loss, centers, centers_update_op


    def get_center_loss2(self, features, labels, alpha, num_classes):
        len_features = features.get_shape()[1]
        centers = tf.get_variable('centers2', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)
        diff = centers_batch - features
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        centers_update_op = tf.scatter_sub(centers, labels, diff)
        return loss, centers, centers_update_op