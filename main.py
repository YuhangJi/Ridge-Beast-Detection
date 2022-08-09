"""2022年8月9日"""
import os
import numpy as np
import tensorflow as tf
from data_process.make_dataset import generateDataset
from yolo.model import Net
from yolo.yolo_loss import YOLOLoss
import datetime
from config import (
    TRAIN_DIR,
    TEST_DIR,
    CATEGORY_NUM,
    EPOCHS,
    LOG_DIR,
    BATCH_SIZE,
    DECAY_RATE,
    DECAY_STEP,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CHANNELS,
    SAVE_MODEL_DIR,
    SAVE_FREQUENCY,
    TEST_FREQUENCY,
    IS_LOAD_WEIGHTS,
    INITIAL_LEARNING_RATE
)
if __name__ == '__main__':
    # Dataset
    train_dataset, train_count = generateDataset(TRAIN_DIR)
    test_dataset, test_count = generateDataset(TEST_DIR)
    # YOLO Net

    net = Net(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        out_channels=3 * (CATEGORY_NUM + 5),
        alpha=0.1
    )
    # YOLO Loss
    yolo_loss = YOLOLoss()
    # Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=DECAY_STEP,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)


    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            yolo_output = net(image_batch, training=True)
            pred_loss = yolo_loss(y_true=label_batch, y_pred=yolo_output)
            regularization_loss = tf.reduce_sum(net.losses)
            net_loss = pred_loss + regularization_loss
        gradients = tape.gradient(net_loss, net.trainable_variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(gradients, net.trainable_variables)
        )

        return pred_loss


    def test_step(image_batch, label_batch):
        yolo_output = net(image_batch, training=True)
        pred_loss = yolo_loss(y_true=label_batch, y_pred=yolo_output)
        regularization_loss = tf.reduce_sum(net.losses)
        net_loss = pred_loss + regularization_loss
        return net_loss


    # LOGS Tracing
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR, current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Load Weights
    if IS_LOAD_WEIGHTS:
        net.load_weights(filepath=SAVE_MODEL_DIR)
        print("Successfully load weights from {} !".format(SAVE_MODEL_DIR))

    # Training
    GLOBAL_STEP = 1
    best_test_loss = [np.inf]
    for epoch in range(1, EPOCHS + 1):
        train_epoch_loss = 0
        train_steps_per_epoch = train_count // BATCH_SIZE
        step = 1
        for batch_data, batch_label1, batch_label2, batch_label3 in train_dataset:
            train_batch_loss = train_step(image_batch=batch_data,
                                          label_batch=[batch_label1, batch_label2, batch_label3])
            if train_batch_loss < 0:
                print("Skip this step due to negative loss.")
                continue
            train_epoch_loss += train_batch_loss
            with summary_writer.as_default():
                tf.summary.scalar('train_batch_loss', train_batch_loss, step=GLOBAL_STEP)
            GLOBAL_STEP += 1

            print(
                "Epoch: {}/{}, Step: {}/{},"
                "batch_loss: {:.5f}, global_steps:{}, lr:{:.9f}.".format(
                    epoch,
                    EPOCHS,
                    step,
                    train_steps_per_epoch,
                    train_batch_loss,
                    GLOBAL_STEP,
                    lr_schedule.__call__(GLOBAL_STEP)
                ))
            step += 1
        with summary_writer.as_default():
            tf.summary.scalar('train_epoch_loss', train_epoch_loss / train_steps_per_epoch, step=epoch)
        print("Epoch: {}/{},Train Average Loss:{:.5f}.".format(
            epoch,
            EPOCHS,
            train_epoch_loss / train_steps_per_epoch
        ))

        # Testing

        if epoch % TEST_FREQUENCY == 0:
            test_loss = 0
            test_steps_per_epoch = test_count // BATCH_SIZE
            for batch_data, batch_label1, batch_label2, batch_label3 in test_dataset:
                loss = test_step(image_batch=batch_data, label_batch=[batch_label1, batch_label2, batch_label3])
                if loss < 0:
                    continue
                test_loss += loss
            test_loss /= test_steps_per_epoch
            with summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss, step=epoch)
            print("Test_loss: {:.5f}".format(test_loss))
            if test_loss < best_test_loss[-1]:
                net.save_weights(SAVE_MODEL_DIR, save_format='tf')
                print(
                    "The best loss on test dataset has declined from {} to {}, and saving model weight to {}."
                        .format(best_test_loss[-1], test_loss, SAVE_MODEL_DIR)
                )
                best_test_loss.append(test_loss)

        if epoch % SAVE_FREQUENCY == 0:
            net.save_weights(filepath=SAVE_MODEL_DIR + "epoch-{}".format(epoch), save_format='tf')
            print("Save Model in {}.".format(SAVE_MODEL_DIR) + "epoch-{}".format(epoch))
