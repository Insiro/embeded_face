import tensorflow as tf

keras = tf.keras
from model import face_mobile
from trainer import ModelTrainer, AccMatrix
from util import convert, PathLoader
from callbacks import SaveModelCallbacak, SaveSummaryCallback

EPOCHS = 100000
BATCH_SIZE = 5
DATA_DIR = "/data/01.datasets/MorphedImage/Female"

pathLoader = PathLoader("./output")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(480, 600),
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(480, 600),
    batch_size=BATCH_SIZE,
)
val_batches = tf.data.experimental.cardinality(val_ds)

test_ds = val_ds.take((2 * val_batches) // 2)
val_ds = val_ds.skip((2 * val_batches) // 2)
class_names = train_ds.class_names

SAVE_PATH = "./output/"
model = None


callbacks = [
    SaveModelCallbacak(path_loader=pathLoader),
    SaveSummaryCallback(path_loader=pathLoader),
]


lr_scheduler = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001, first_decay_steps=50, t_mul=2, m_mul=0.9
)

with tf.device("/gpu:0"):
    model = face_mobile(510)
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)

    train_loss = keras.metrics.Mean(name="train_loss")
    train_acc = keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    test_loss = keras.metrics.Mean(name="test_loss")
    test_acc = keras.metrics.SparseCategoricalAccuracy(name="test_acc")
    trainer = ModelTrainer(
        model,
        None,
        loss_function=loss_function,
        optimizer=optimizer,
        acc_matrix=AccMatrix(
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        ),
    )
    trainer.add_callback(callbacks)
    trainer.train(EPOCHS, train_ds, test_ds)

    image, labels = tuple(zip(*val_ds))
    result = model.evaluate(image, labels)
    with open("./result.txt", "w") as ff:
        ff.write(result)


model.save("./output.h5")
open("./test.tflite", "wb").write(convert(model))
