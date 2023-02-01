import tensorflow as tf


def load_data(config, dir="data", split_val=False):
    data_path = config["dir"][dir]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=config["valid_split"],
        subset="training",
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=config["valid_split"],
        subset="validation",
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    test_ds = None
    if split_val:
        val_batches = tf.data.experimental.cardinality(val_ds)

        test_ds = val_ds.take((2 * val_batches) // 2)
        val_ds = val_ds.skip((2 * val_batches) // 2)
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names


def load_data2(config):
    dir = config["dir"]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir["data"],
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir["test"],
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, None, class_names
