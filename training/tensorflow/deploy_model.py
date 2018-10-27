
import warnings
import argparse
import PATH
import models.ImageData as ImgData
import models.model as mod
import models.cnn_models as fe
import models.util as util



def restore_model(p=PATH.paths, input_shape=(128, 128, 3), ex=fe.CNN, classes=None, learning_rate=1e-5, rewrite=False, epoch=1452):
    model = mod.ClassificationModel(path=p["train_results"], input_shape=input_shape, classnames=classes,
                                    extractor=ex, training=False, learning_rate=learning_rate, dropout=0.0,
                                    rewrite=rewrite)
    model.inference()
    restore_path = model.save_dir + "net-" + str(epoch)
    model.load(path=restore_path)
    return model


def load_data(dataset, p=PATH.paths, input_shape=(128, 128, 3)):
    if dataset == "dataset":
        data = ImgData.CancerData(p["dataset"], input_shape)
    else:
        data = ImgData.CancerData(p["cifar10"], input_shape)
    data.load()
    # check_data(data)
    return data

def test_img(img,model):
    x_batch = img #todo reshape to batchsize 1
    result = model.predict(x_batch)
    return result




if __name__ == "__main__":
        input_shape = (256, 256, 3)
        print(PATH.paths)

        data = load_data("dataset", p=PATH.paths, input_shape=input_shape)
        model = restore_model(p=PATH.paths, input_shape=input_shape, ex=fe.CNN3x3, classes=data.get_classnames(), learning_rate=1e-4, rewrite=False)
