"""
Created on 1/13/19
@author: MRChou

Scenario: provide simple multi-layer MLP for VSB power dataset
"""

from functools import partial

import pandas
import tensorflow as tf
from tensorflow.metrics import recall_at_thresholds as recall
from tensorflow.metrics import precision_at_thresholds as precision

from TF_Utils.Models.MLP import MLP
from TS_Utils.seriesObjs import ParquetArray

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True

LABEL = '/rawdata/VSB_Power/metadata_train.csv'
POSCOUNT = 525
NEGCOUNT = 8187


def _parquetfilter(target):
    label = pandas.read_csv(LABEL)
    index = label[label['target'] == target].index
    return index


def mlp_input(file, batch, epoch, skip=None, take=None, upsample=True):
    # note: 525 fault signals over 8712
    pos = ParquetArray(file)[_parquetfilter(1)].to_dataset()
    neg = ParquetArray(file)[_parquetfilter(0)].to_dataset()

    count = NEGCOUNT
    if skip and take:
        raise ValueError("Can have only 'skip' or 'take', received both.")
    elif skip:
        pos = pos.skip(int(POSCOUNT*skip))
        neg = neg.skip(int(NEGCOUNT*skip))
        count = count - int(NEGCOUNT*skip)
    elif take:
        pos = pos.take(int(POSCOUNT*take))
        neg = neg.take(int(NEGCOUNT*take))
        count = int(NEGCOUNT*take)

    if upsample:
        pos = pos.repeat()
        neg = neg.repeat()
        ds = tf.contrib.data.sample_from_datasets([pos, neg]).take(2*count)
    else:
        ds = pos.concatenate(neg)
        ds = ds.shuffle(buffer_size=count)

    ds = ds.batch(batch)
    ds = ds.repeat(epoch)
    ds = ds.make_one_shot_iterator()
    srsid, srs = ds.get_next()

    label = tf.convert_to_tensor(pandas.read_csv(LABEL)['target'].values,
                                 dtype=tf.float32
                                 )
    label = tf.gather(label, srsid)
    label = tf.reshape(label, [-1, 1])
    srs = tf.reshape(srs, [-1, 800000])
    mean, variance = tf.nn.moments(srs, axes=[0, 1])
    srs = (srs - mean) / tf.sqrt(variance)
    return {'srsid': srsid, 'srs': srs}, label


def model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    model = MLP(hidden_units=[256, 512, 256],
                activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu],
                n_classes=1,
                batchnorm=True)

    feature_vector = features['srs']
    output = model(feature_vector, training=istrain)

    if istrain or iseval:
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                               logits=output)
        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, global_step=globalstep)
    else:
        loss = None
        train_op = None

    if ispredict or iseval:  # i.e. not in train mode
        predictions = {'probability': tf.nn.sigmoid(output),
                       'srs_id': features['srsid']}
    else:
        predictions = None

    if iseval:
        eval_metric = {'Recalls': recall(labels, predictions['probability'],
                                         (0.1, 0.3, 0.5, 0.7, 0.9)),
                       'Precision': precision(labels, predictions['probability'],
                                              (0.1, 0.3, 0.5, 0.7, 0.9))}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def script_train():
    file = '/rawdata/VSB_Power/train.parquet'
    model_dir = '/archive/VSB/models/MLP'
    train_input_fn = partial(mlp_input, file=file, batch=1, epoch=1, upsample=True, take=0.8)
    eval_input_fn = partial(mlp_input, file=file, batch=1, epoch=1, upsample=False, skip=0.8)

    lr = 0.01
    momentum = 0.3
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    for step in range(100):
        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       config=config,
                                       params={'sgdlr': lr,
                                               'sgdmomt': momentum})

        print(model.evaluate(input_fn=eval_input_fn))
        model.train(input_fn=train_input_fn)
        tf.reset_default_graph()
        lr = lr*0.9


def test():
    file = '/rawdata/VSB_Power/train.parquet'
    train_input_fn = partial(mlp_input, file=file, batch=1, epoch=1, upsample=False, take=0.8)
    inputs, label = train_input_fn()
    with tf.Session() as sess:
        for _ in range(10):
            print(sess.run([inputs, label]))


if __name__ == "__main__":
    # test()
    script_train()
