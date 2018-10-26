import warnings
import argparse
import PATH
import training.tensorflow.models.ImageData as ImgData
import training.tensorflow.models.model as mod
import training.tensorflow.models.cnn_models as fe
import training.tensorflow.models.util as util

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


def check_data(data):
    print(data.size_train())
    x_batch, y_batch = data.random_batch(10, mode="train")
    util.show_batch(x_batch=x_batch, y_batch=y_batch)
    print(data.size_test())
    x_batch, y_batch = data.random_batch(10, mode="test")
    util.show_batch(x_batch=x_batch, y_batch=y_batch)


def load_data(dataset, p=PATH.paths, input_shape=(128, 128, 3)):
    if dataset == "dataset":
        data = ImgData.CancerData(p["dataset"], input_shape)
    else:
        data = ImgData.CancerData(p["cifar10"], input_shape)
    data.load()
    check_data(data)
    return data


def restore_model(p=PATH.paths, input_shape=(128, 128, 3), ex=fe.CNN, classes=None, learning_rate=1e-5, rewrite=False, epoch=1562):
    model = mod.ClassificationModel(path=p["train_results"], input_shape=input_shape, classnames=classes,
                                    extractor=ex, training=False, learning_rate=learning_rate, dropout=0.0,
                                    rewrite=rewrite)
    model.inference()
    restore_path = model.save_dir + "net-" + str(epoch)
    model.load(path=restore_path)
    return model


def test_batch(data, models=[], batch_size=32):
    x_batch, y_batch = data.random_batch(batch_size, mode="train")
    x_batch_test, y_batch_test = data.random_batch(batch_size, mode="test")

    for index, m in enumerate(models):
        m.evaluate(x_batch, y_batch, mode="train")
        m.evaluate(x_batch_test, y_batch_test, mode="test")
    print('Testing completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', help='Paths to use')
    args = parser.parse_args()
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            input_shape = (256, 256, 3)
            print(PATH.paths)
            data = load_data("dataset", p=PATH.paths)
            print(data.get_classnames())
            models = []

            learningrate = [1e-4] #, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6]

            for l in learningrate:
                model = restore_model(p=PATH.paths, input_shape=input_shape, ex=fe.CNN_Resnet, classes=data.get_classnames(), learning_rate=l, rewrite=False)
                models.append(model)
                test_batch(data=data, models=models, batch_size=20)
                model = None
                models = []
