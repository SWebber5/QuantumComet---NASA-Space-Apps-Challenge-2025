import tensorflow as tf
import numpy as np

def retrieve_data(tfrecord_file):
    feature_description = {
        "global_view": tf.io.FixedLenFeature([2001], tf.float32),
        "local_view": tf.io.FixedLenFeature([201], tf.float32),
        "av_training_set": tf.io.FixedLenFeature([], tf.string),
        "kepid": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_tfrecord(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        
        # Convert label to integer
        label = tf.cond(tf.equal(example["av_training_set"], tf.constant("PC")),
                        lambda: tf.constant(1, dtype=tf.int64), # Represents a planet candidate
                        lambda: tf.constant(0, dtype=tf.int64)) # Represents a false positive
        
        # Return (global, local) as features
        return (example["global_view"], example["local_view"]), label

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord)
    return dataset

def train_data():
    train_datasets = []
    for i in range(8):
        train_datasets.append(retrieve_data(f"train-0000{i}-of-00008"))

    Xg_train, Xl_train, label_train = [], [], []
    for dataset in train_datasets:
        for (global_view, local_view), label in dataset:
            Xg_train.append(global_view.numpy())
            Xl_train.append(local_view.numpy())
            label_train.append(label.numpy())

    Xg_train = np.array(Xg_train)
    Xl_train = np.array(Xl_train)
    label_train = np.array(label_train)
    return Xg_train, Xl_train, label_train

def validation_data():
    valid_dataset = retrieve_data(f"val-00000-of-00001")

    Xg_valid, Xl_valid, label_valid = [], [], []
    for (global_view, local_view), label in valid_dataset:
        Xg_valid.append(global_view.numpy())
        Xl_valid.append(local_view.numpy())
        label_valid.append(label.numpy())

    Xg_valid = np.array(Xg_valid)
    Xl_valid = np.array(Xl_valid)
    label_valid = np.array(label_valid)
    return Xg_valid, Xl_valid, label_valid


def test_data():
    test_dataset = retrieve_data(f"test-00000-of-00001")

    Xg_test, Xl_test, label_test = [], [], []
    for (global_view, local_view), label in test_dataset:
        Xg_test.append(global_view.numpy())
        Xl_test.append(local_view.numpy())
        label_test.append(label.numpy())

    Xg_test = np.array(Xg_test)
    Xl_test = np.array(Xl_test)
    label_test = np.array(label_test)
    return Xg_test, Xl_test, label_test