import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from importlib import import_module
import os
from sklearn.model_selection import train_test_split
import basic_arch

def load_meas(master_dir, subdir, file_name, debug=True):
    meas = import_module(f'{master_dir}.{subdir}.{file_name}')
    red = getattr(meas, "red")
    blue = getattr(meas, "blue")
    times = getattr(meas, "times")
    spo2 = getattr(meas, "spo2")
    # todo: add support in green

    red = np.array(red)
    blue = np.array(blue)
    times = np.array([(t - times[0]) / (10 ** 9) for t in times])
    if debug:
        print(f'load {file_name}, red: {red.shape}, blue: {blue.shape}, time: {times.shape}')
    return red, blue, times, spo2


def interpolate(red, blue, times):
    red_time_diff = len(times) - len(red)
    blue_time_diff = len(times) - len(blue)

    f_red = interp.CubicSpline(time_points[signal_cut + red_time_diff:], red[signal_cut:])
    f_blue = interp.CubicSpline(time_points[signal_cut + blue_time_diff:], blue[signal_cut:])

    resembled_red = f_red(spaced_time_points[int(signal_cut * 2):len(spaced_time_points) - 120])
    resembled_blue = f_blue(spaced_time_points[int(signal_cut * 2):len(spaced_time_points) - 120])

    assert resembled_red.shape == resembled_blue.shape, f'red and blue should have the same shape after interpolation'
    return resembled_red, resembled_blue


def plot_graph(red, blue):
    """ to plot to
    # fig = plt.figure(figsize=(20, 15))
    #
    # red_plt = fig.add_subplot(311)
    # blue_plt = fig.add_subplot(312)
    # delta_t = fig.add_subplot(313)
    #
    # red_plt.plot(time_points[signal_cut + red_time_diff:], red_array[signal_cut:])
    # blue_plt.plot(time_points[signal_cut + blue_time_diff:], blue_array[signal_cut:])
    # delta_t.scatter(range(time_points.size - 1), np.diff(time_points))
    """
    cut_points = spaced_time_points[int(signal_cut * 2):len(spaced_time_points) - 120]

    fig = plt.figure(figsize=(20, 10))
    red_re = fig.add_subplot(211)
    blue_re = fig.add_subplot(212)

    red_re.plot(cut_points, red)
    blue_re.plot(cut_points, blue)
    plt.show()


def split_to_batches(red, blue, spo2, batches_num):
    db_x = np.zeros((batches_num, channels, signal_len // batches_num))
    db_y = np.full(batches_num, spo2)
    for batch in range(batches_num):
        db_x[batch, 0] = red[(signal_len // batches_num) * batch:(signal_len // batches_num) * (batch + 1)]
        db_x[batch, 1] = blue[(signal_len // batches_num) * batch:(signal_len // batches_num) * (batch + 1)]
    return db_x, db_y


signal_cut = 70
spaced_time_points = np.arange(0, 30, 1 / 30)
batches = 5
signal_len = len(spaced_time_points) - 120 - signal_cut * 2
channels = 2
if __name__ == '__main__':
    measurements_dir = 'Measurements'
    measurements_subdir = 'RAW_100X100'
    meas_names = os.listdir(f'{measurements_dir}/{measurements_subdir}')
    meas_names = [file[:-3] for file in meas_names if file[-2:] == "py"]
    database_x = np.zeros(shape=(batches * len(meas_names), channels, signal_len // batches))
    database_y = np.zeros(shape=(batches * len(meas_names)))
    for i, meas_name in enumerate(meas_names):
        red_array, blue_array, time_points, spo2_g = load_meas(measurements_dir, measurements_subdir, meas_name)
        red_intrp, blue_intrp = interpolate(red_array, blue_array, time_points)
        # plot_graph(red_intrp, blue_intrp)

        data_x, data_y = split_to_batches(red_intrp, blue_intrp, spo2_g, batches)
        database_x[i * batches: (i + 1) * batches] = data_x
        database_y[i * batches: (i + 1) * batches] = data_y
    x_train, y_train, x_val, y_val = train_test_split(database_x, database_y, test_size=0.33, random_state=42)

    net = basic_arch.build_net_1(signal_len // batches)
    net.fit(x_train, y_train, epochs=10, batch_size=5)
    print(net.evaluate(x_val, y_val))
