import cv2
#import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')


def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def add_summaries():
    """Attach all tf trainable_variables to tensorboard."""
    with tf.name_scope('summaries'):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
            mean = tf.reduce_mean(var)
            tf.summary.scalar(var.op.name + 'mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(var.op.name + 'stddev', stddev)
            tf.summary.scalar(var.op.name + 'max', tf.reduce_max(var))
            tf.summary.scalar(var.op.name + 'min', tf.reduce_min(var))
            tf.summary.histogram(var.op.name + 'histogram', var)


def show_batch(x_batch, y_batch):
    print("SHOW BATCH")
    for i in range(0, len(x_batch) - 1):
        cv2.namedWindow('x_batch', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('x_batch', 300, 300)
        cv2.imshow('x_batch', x_batch[i])
        print(y_batch[i], x_batch[i].shape)
        # plt.hist(x_batch[i].ravel(), 255, [1, 255])
        # plt.show()
        cv2.waitKey(0)


def show_results(x, y, distances):
    x1, x2, = x
    for i in range(len(x1)):
        output = create_output(x1[i], x2[i], 'data', y[i][0], distances[i][0])
        cv2.imshow('Result', output)
        print('class: ', y[i][0], ' distancepred: ', distances[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_predictions(y, y_pred, string):
    y_preds = np.resize(y_pred, (len(y_pred), 1))
    same = y_preds[y > 0.5]
    different = y_preds[y < 0.5]
    plt.hist(different, np.linspace(0, y_preds.max(), 15), alpha=0.5, label='different')
    plt.hist(same, np.linspace(0, y_preds.max(), 15), alpha=0.5, label='same')
    plt.legend(loc='upper right')
    plt.savefig(string)
    plt.clf()


def mk_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def mk_dirs(directory):
    mk_dir(directory)
    model_folder = os.path.join(directory, 'net')
    hist_folder = os.path.join(directory, 'hist')
    log_folder = os.path.join(directory, 'logs')
    log_train = os.path.join(log_folder, 'train')
    log_test = os.path.join(log_folder, 'test')
    mk_dir(model_folder)
    mk_dir(hist_folder)
    mk_dir(log_folder)
    mk_dir(log_train)
    mk_dir(log_test)


def clear_dir(directory):
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    mk_dirs(directory)


def create_output(sig1, sig2, path, actual, distance):
    accepted = cv2.imread(os.path.join(path, 'accepted.PNG'))
    accepted = cv2.resize(accepted, (200, 200))
    rejected = cv2.imread(os.path.join(path, 'rejected.png'))
    rejected = cv2.resize(rejected, (200, 200))
    check = cv2.imread(os.path.join(path, 'tobechecked.png'))
    check = cv2.resize(check, (200, 200))

    rows, cols, _ = accepted.shape

    resized_height = 208
    resized_width = 416
    ycord = 40

    img1 = cv2.resize(sig1, (resized_width, resized_height), cv2.INTER_NEAREST)
    img2 = cv2.resize(sig2, (resized_width, resized_height), cv2.INTER_NEAREST)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    img1 = cv2.bitwise_not(img1)
    img2 = cv2.bitwise_not(img2)

    vis = np.zeros((520, 882, 3), np.uint8)
    vis[vis > -1] = 255

    height1 = ycord + resized_height

    vis[ycord:height1, 20:436] = img1
    vis[ycord:height1, 456:872] = img2
    if actual == 1.0:
        vis[300:500, 108:308] = accepted # 228 middle
    else:
        vis[300:500, 108:308] = rejected

    if distance < 0.4:
        vis[300:500, 570:770] = accepted
    elif 0.4 <= distance <= 0.6:
        vis[300:500, 570:770] = check
    else:
        vis[300:500, 570:770] = rejected

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    font_color = (54, 63, 20)  # R 20 G 63 B 54
    line_type = 1

    cv2.putText(vis, 'First Signature', (120, 36), font, font_scale, font_color, line_type)
    cv2.putText(vis, 'Second Signature', (520, 36), font, font_scale, font_color, line_type)
    cv2.putText(vis, 'Actual', (150, 295), font, font_scale, font_color, line_type)
    cv2.putText(vis, 'Predicted', (590, 295), font, font_scale, font_color, line_type)
    cv2.putText(vis, 'Distance', (370, 295), font, font_scale, font_color, line_type)
    cv2.putText(vis, str(round(distance, 2)), (390, 380), font, font_scale, font_color, line_type)

    return vis


def create_output_classification(img, classes, y, y_logits, string):
    barlist = plt.bar(classes, y_logits)
    for i in range(len(classes)):
        barlist[i].set_color("r")

    barlist[y].set_color("g")

    # custom_x_ticks = classes
    # plt.xticks(x, custom_x_ticks)
    plt.ylim([0,1])
    plt.title("Logits")
    plt.savefig(string)
    plt.close()


    resized_height = 300
    resized_width = 300
    padding = 50
    img = img*255
    img = cv2.resize(img, (resized_width, resized_height), cv2.INTER_NEAREST)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    output = np.full((resized_height+2*padding, 2*resized_width+3*padding, 3), 255, np.uint8)
    output[padding:padding+resized_height, padding:padding+resized_width] = img
    chart = cv2.imread(string)
    chart = cv2.resize(chart, (resized_width, resized_height), cv2.INTER_NEAREST)
    output[padding:padding+resized_height, padding+resized_width+padding:2*padding+2*resized_width] = chart

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    font_color = (54, 63, 20)  # R 20 G 63 B 54
    line_type = 1
    cv2.putText(output, 'Results', (padding+resized_width, int(padding/2)), font, font_scale, font_color, line_type)
    # cv2.putText(output, 'Input', (padding, padding), font, font_scale, font_color, line_type)
    # cv2.putText(output, 'Output', (2*padding+resized_width, padding), font, font_scale, font_color, line_type)
    cv2.imwrite(string, output)
