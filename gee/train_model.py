import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import Sequence

from utils import mask_unlabeled_values, make_dataset
from models import unet
import feature_spec

features_dict = feature_spec.features_dict()
bands = feature_spec.bands()
features = feature_spec.features()

N_CLASSES = 3 


bands = list(features_dict.keys())[:-1]
features = list(features_dict.keys())


def m_acc(y_true, y_pred):
    y_true_sum = tf.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    acc = K.mean(K.equal(y_pred_masked, y_true_masked))
    return acc


class StreamingF1Score(Metric):
    '''
    Stateful metric class to accumulate confusion matrices
    and calculate f1 score from them.
    focus_on_class only reports f1 score for the class labeled with
    the integer that is passed in.
    '''

    def __init__(self, name='f1', num_classes=2, 
            focus_on_class=None, **kwargs):
        super(StreamingF1Score, self).__init__(name=name, **kwargs)
        self.cmats = self.add_weight(name='cmats', shape=(num_classes, 
            num_classes), dtype=tf.float32, initializer='zeros')
        self.num_classes = num_classes
        self.focus_on_class = focus_on_class


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = mask_unlabeled_values(y_true, y_pred)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        self.cmats.assign_add(tf.cast(tf.math.confusion_matrix(y_true,
           y_pred, num_classes=self.num_classes), tf.float32))
        

    def reset_states(self):
        tf.print(self.cmats, output_stream='file://cmat.out')
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes),
            dtype=np.float32)) for v in self.variables])


    def result(self):
        f1 = self._result(self.cmats)
        if self.focus_on_class is not None:
            tf.print(f1[self.focus_on_class], output_stream='file://f1irr.out')
            return f1[self.focus_on_class]
        else:
            return tf.reduce_mean(f1)


    def _result(self, cmats):
        # returns diagonals of shape (num_classes,).
        prec = cmats / tf.reduce_sum(cmats, axis=1)
        rec = cmats / tf.reduce_sum(cmats, axis=0)
        prec = tf.linalg.tensor_diag_part(prec)
        rec = tf.linalg.tensor_diag_part(rec)
        f1 = 2*(prec*rec)/(rec+prec)
        return f1



def lr_schedule(epoch):
    lr = 1e-3
    rlr = 1e-3
    if epoch > 25:
        rlr = lr / 2
    if epoch > 50:
        rlr = lr / 4
    if epoch > 75:
        rlr = lr / 6
    if epoch > 100:
        rlr = lr / 8
    if epoch > 125:
        rlr = lr / 16
    if epoch > 150:
        rlr = lr / 32
    tf.summary.scalar('learning rate', data=rlr, step=epoch)
    return rlr

if __name__ == '__main__':
    test_root = '/home/thomas/ee-test/data/training_data/test/'
    train_root = test_root

    sf1 = StreamingF1Score(num_classes=3, focus_on_class=0)
    model = unet((None, None, 36), n_classes=N_CLASSES, initial_exp=4)
    model.compile(Adam(1e-3), loss='categorical_crossentropy',
            metrics=[m_acc, sf1])

    train = make_dataset(train_root)
    test = make_dataset(test_root)

    model_out_path = './models/'

    if False:
        tb_path = os.path.join(model_out_path, 'logs')
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)
        model_out_path = os.path.join(model_out_path, 'model-{val_f1:.3f}-{val_m_acc:.3f}.h5')
        lr = LearningRateScheduler(lr_schedule, verbose=True)
        chpt = ModelCheckpoint(model_out_path, 
                save_best_only=True, verbose=True, 
                monitor='val_f1', mode='max') 
        
        tb = TensorBoard(log_dir=tb_path,
                         update_freq='epoch',
                         write_images=False)


        model.fit(train,
                  steps_per_epoch=175,
                  epochs=300,
                  validation_data=test,
                  validation_steps=75,
                  callbacks=[chpt, lr, tb])
        model.save("fully_trained.h5")

              
    else:
        model.load_weights('./models/model-0.980-0.919.h5')
        total = test.unbatch().batch(1)
        for batch in total:
            preds = model.predict(batch)
            for i in range(preds.shape[0]):
                pred = preds[i]
                label = batch[1][i].numpy().squeeze()
                fig, ax = plt.subplots(ncols=3)
                bad = np.where(np.sum(label, axis=-1) == 0)
                label = label.astype(np.float32)
                label[bad] = np.nan
                ax[0].imshow(pred)
                ax[1].imshow(label)
                ax[2].imshow(batch[0][i, :, :, :3])
                #plt.suptitle(m_acc(label, pred))
                plt.show()
