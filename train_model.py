import os
print(os.getcwd())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import argparse
import PATH
import models.ImageData as ImgData
import models.model as mod
import models.cnn_models as fe
import numpy as np
import models.util as util

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


def check_data(data):
    print(data.size_train())
    x_batch, y_batch = data.random_batch(10, mode="train")
    util.show_batch(x_batch=x_batch, y_batch=y_batch)

    x_batch, y_batch = data.random_batch(10, mode="test")
    util.show_batch(x_batch=x_batch, y_batch=y_batch)


def load_data(dataset, p=PATH.paths, input_shape=(128, 128, 3)):
    if dataset == "dataset":
        data = ImgData.CancerData(p["dataset"], input_shape)

    if dataset == "cifar10":
        data = ImgData.Cifar10Data(p["cifar10"], input_shape)

    data.load()
    return data


def load_model(p=PATH.paths, input_shape=(128, 128, 3), ex=fe.CNN, classes=None, learning_rate=1e-5, rewrite=False):
    model = mod.ClassificationModel(path=p["train_results"], input_shape=input_shape, classnames=classes,
                                    extractor=ex, training=True, learning_rate=learning_rate, dropout=0.0,
                                    rewrite=rewrite)
    model.inference()
    return model


def training(data, models=[], epochs=512, batch_size=32):
    iter_size = int(data.size_train() / batch_size)
    loss_histories = [dict() for m in enumerate(models)]
    loss_delta = 1e-15

    print("Batches per epoch:", iter_size)
    for epoch in range(epochs):
        models_epoch_loss = np.zeros(len(models), dtype=np.float32)

        for i in range(iter_size):
            x_batch, y_batch = data.random_batch(batch_size, mode="train")
            for index, m in enumerate(models):
                _, loss_train = m.sess.run([m.model["optimizer"], m.model["loss"]],
                                           feed_dict={m.model["x"]: x_batch, m.model["y"]: y_batch})
                models_epoch_loss[index] += loss_train
                m.global_step.assign_add(1)
                print('Model:', m.name, 'Epoch:', epoch, ' Iter:', i, ' Loss:', loss_train,
                      'Progress: {:2.1%}'.format((i + 1) / iter_size))

                if i % 20 == 0:
                    # Create Summaries
                    x_batch, y_batch = data.random_batch(50, mode="train")
                    x_batch_test, y_batch_test = data.random_batch(50, mode="test")
                    m.summary(x_batch, y_batch, mode="train")
                    m.summary(x_batch_test, y_batch_test, mode="test")

        for index, m in enumerate(models):
            loss_histories[index][str(epoch)] = models_epoch_loss[index]
            print('Model:', m.name, 'Epoch:', epoch, 'Epoch Loss:', models_epoch_loss[index])

            if epoch % 32 == 0:
                m.save()
            if epoch > 3:
                if loss_histories[index][str(epoch - 1)] - loss_histories[index][str(epoch)] < loss_delta and \
                                loss_histories[index][str(epoch - 2)] - loss_histories[index][str(epoch)] < loss_delta and \
                                loss_histories[index][str(epoch - 3)] - loss_histories[index][str(epoch)] < loss_delta:
                    print("Early stopping of ", m.name, " - Creating final summary and removing from training pipeline")
                    m.summary(x_batch, y_batch, mode="train")
                    m.summary(x_batch_test, y_batch_test, mode="test")
                    m.save()
                    m.evaluate(x_batch, y_batch, mode="train")
                    m.evaluate(x_batch_test, y_batch_test, mode="test")
                    models.remove(m)
                    '''
                    # At the last step, add the incorrectly classified images to TensorBoard
                    if False:  # TODO fix method for adding images to tensorboard
                        pred, pred_bool = m.sess.run([m.model["predictions_op"], m.model["correct_prediction_op"]],
                                                     feed_dict={m.model["x"]: x_batch, m.model["y"]: y_batch})
                        image_summary_op, wrong_images = m.get_misclassified_images(pred_bool, pred)
                        image_summary = m.sess.run(image_summary_op, feed_dict={m.model["tb_images"]: wrong_images})
                        m.model["merged"][1].add_summary(image_summary, i)
                    '''



    for m in models:
        m.save()
    print('Training completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', help='Paths to use')
    parser.add_argument('--e', help='epochs')
    parser.add_argument('--b', help='batchsize')
    parser.add_argument('--s', help='save')
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        input_shape = (256, 256, 3)
        print(PATH.paths)

        data = load_data("dataset", p=PATH.paths, input_shape=input_shape)
        models = []
        learningrate = [1e-4] #, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6]

        for l in learningrate:
            model = load_model(p=PATH.paths, input_shape=input_shape, ex=fe.inception_v3_base, classes=data.get_classnames(), learning_rate=l, rewrite=True)
            model2 = load_model(p=PATH.paths, input_shape=input_shape, ex=fe.CNN_Resnet, classes=data.get_classnames(), learning_rate=l, rewrite=True)
            model3 = load_model(p=PATH.paths, input_shape=input_shape, ex=fe.CNN3x3, classes=data.get_classnames(), learning_rate=l, rewrite=True)
            models.append(model)
            models.append(model2)
            models.append(model3)
            training(data=data, models=models, epochs=128, batch_size=32)

            model = None
            model2 = None
            model3 = None
            models = []

        # model = load_model(p=PATH.paths, input_shape=input_shape, ex=fe.CNN_Resnet, classes=data.get_classes(), rewrite=True)
        # models.append(model)
        # model = load_model(p=PATH.paths, input_shape=input_shape, ex=fe.CNN3x3, classes=data.get_classes(), rewrite=True)
        # models.append(model)

        # Sequential Training
        # for model in models:
        #    training(data=data, models=[model], epochs=512, batch_size=16)

