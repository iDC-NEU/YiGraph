import time
from collections import defaultdict
import statistics


class TimerCtx:

    def __init__(self, timer, key, verbose=False):
        self.timer = timer
        self.key = key
        self.verbose = verbose

    def __enter__(self):
        self.timer.start_time_dict[self.key] = time.time()
        return self

    def __exit__(self, type, value, traceback):
        d = time.time() - self.timer.start_time_dict[self.key]
        self.timer.duration_dict[self.key].append(d)
        self.timer.count_dict[self.key] += 1
        if self.verbose:
            print(f"{self.key} cost {d:.3f}s")


class Timer:

    def __init__(self, verbose=False):
        self.start_time_dict = {}
        self.duration_dict = defaultdict(list)
        self.count_dict = defaultdict(int)
        self.verbose = verbose

    def summary(self, skip=3):
        avg_dict = {}
        std_dict = {}
        for key in self.duration_dict:
            data = self.duration_dict[key][skip:]
            # print(data)
            avg_dict[key], std_dict[key] = statistics.mean(
                data), statistics.stdev(data) if len(data) > 1 else 0
        s = '\ntimer summary:\n' + "\n".join(
            "%6.2fs %6.2fs %5d %s" %
            (avg_dict[key], std_dict[key], self.count_dict[key], key)
            for key in self.duration_dict)
        return s

    def summary_dict(self, skip=3):
        avg_std_dist = {}
        for key in self.duration_dict:
            data = self.duration_dict[key][skip:]
            avg_std_dist[f'{key}_avg_std'] = '%.2fs %.2fs' % (statistics.mean(
                data), statistics.stdev(data) if len(data) > 1 else 0)
        return avg_std_dist

    def detail(self):
        avg_dict = {}
        std_dict = {}
        detail_dict = {}
        for key in self.duration_dict:
            data = self.duration_dict[key]
            avg_dict[key], std_dict[key] = statistics.mean(
                data), statistics.stdev(data) if len(data) > 1 else 0
            detail_dict[key] = ' '.join("%6.2f" % x for x in data)
        s = '\ntimer summary:\n' + "\n".join(
            "%6.2fs %6.2fs %5d %s \ndetail: %s \n--------------" %
            (avg_dict[key], std_dict[key], self.count_dict[key], key,
             detail_dict[key]) for key in self.duration_dict)
        return s

    def add_duration_list(self, key, values):
        if key in self.duration_dict:
            print(f"{key} in timer.duration_dict, it will overwrite it!")
        self.duration_dict[key] = values
        self.count_dict[key] = len(values)

    def add(self, key, value):
        self.duration_dict[key].append(value)
        self.count_dict[key] += 1

    def timing(self, key):
        return TimerCtx(self, key, self.verbose)

    def start(self, key):
        self.start_time_dict[key] = time.time()
        return self.start_time_dict[key]

    def stop(self, key):
        d = time.time() - self.start_time_dict[key]
        self.duration_dict[key].append(d)
        self.count_dict[key] += 1
        return


if __name__ == '__main__':
    timer = Timer()

    with timer.timing('task1'):
        time.sleep(1)

    with timer.timing('task1'):
        time.sleep(2)

    with timer.timing('task1'):
        time.sleep(1)

    timer.start('task2')
    time.sleep(2)
    timer.stop('task2')

    timer.start('task2')
    time.sleep(1)
    timer.stop('task2')

    print(timer.summary())
    print(timer.detail())
