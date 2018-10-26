def test_metrics():
    import models.metrics as metrics
    import numpy as np
    import tensorflow as tf

    # Unit Tests
    # 3 output classes
    x = np.array([[10, -1, 20.2],
                  [-1000000, 1000, -1000000],
                  [0, 0, -10000],
                  [1000, -10000, -10000]])

    x1 = np.array([[-1000000, -1000000, 1.1],
                   [-1000000, 1, -1000000],
                   [1, -1000000, -1000000],
                   [1, -1000001, -1000000]])

    y = np.array([2, 1, 0, 0])

    # Bad results
    y1 = np.array([1, 0, 0, 2])


    with tf.Session() as sess:
        print("xy")
        print(sess.run(metrics.sparse_xentropy_loss(x, y)))
        print("xy1")
        print(sess.run(metrics.sparse_xentropy_loss(x, y1)))
        print("x1y")
        print(sess.run(metrics.sparse_xentropy_loss(x1, y)))
        print("x1y1")
        print(sess.run(metrics.sparse_xentropy_loss(x1, y1)))



def test_image_data():
    import PATH as Path
    import models.ImageData as ImgData

    shape = (155, 220, 1)
    paths = Path.paths_local
    data = ImgData.Data(paths[0], shape)
    data.load_steel()
    x, y = data.random_batch(50, mode="train")
    print(x.shape)
    print(y)
    ImgData.show_batch(x, y)
    x, y = data.random_batch(20, mode="test")
    ImgData.show_batch(x, y)


test_metrics()
#test_image_data()
