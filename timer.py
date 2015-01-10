import time
from utilities import *

enabled = True

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def tic(self):
        self.start_time = time.time()

    def time(self):
        return time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

        return self.total_time

    def elapsed(self):
        print 'Elapsed: %s.' % (time.time() - self.start_time)

    def compute_flops(self, num_operations):
        print 'Speed: %f Gflops.' % (num_operations/self.total_time*1e-9)


class Profiler(dict):
    def __init__(self):
        self.active = False
        self.total_time = 0.
        self.session = 1

    def update(self, function, total_time):
        if self.active:
            if function in self.keys():
                self[function][0] += 1
                self[function][1] += total_time

            else:
                self[function] = [1, total_time]

    def update_total_time(self):
        time_now = time.time()
        self.total_time += time_now - self.time_last_update
        self.time_last_update = time_now

    def start(self):
        assert self.active == False, 'Cannot start profiler - it has already been started.'
        self.active = True
        self.time_last_update = time.time()
        self.session += 1

    def pause(self):
        self.update_total_time()
        assert self.active == True, 'Cannot pause profiler - it has already been paused.'
        self.active = False

    def get_ordered_times(self):
        if self.active:
            self.update_total_time()

        if len(self) == 0:
            print 'No routines found in profiler record.'

        else:
            tuple_representation = [(name, values[0], values[1]) for name, values in self.iteritems()]
            ordered = sorted(tuple_representation, key = lambda row: row[2], reverse = True)

            total_time = sum([item[2] for item in ordered])
            ordered = [(row[0], row[2]/self.total_time*100., row[2], row[1], row[2]/row[1], row[2]/self.session) for row in ordered]

            print '='*10 + ' Routine Profiler Diagnostics ' + '='*10
            print 'Total time profiler was active for is %.3gs.\n' % self.total_time
            print '%d sessions were recorded in this time.\n' % self.session
            print 'The profiled routines took %.3g%% of this time.' % (total_time/self.total_time*100.)

            titles = ['routine'] + ['%% of total'] + ['total time (s)'] + ['calls'] + ['time/call (s)'] + ['time/session (s)']
            str_lengths = [max(12, max([len(row[0]) for row in ordered]) + 4), 16, 18, 13, 17, 17]

            print_table(titles, str_lengths, ordered)
        

def profile_routine(function):
    if not enabled:
        def profiled_routine(*args, **kwargs):
            output = function(*args, **kwargs)
            return output 

    else:
        def profiled_routine(*args, **kwargs):
            start_time = stopwatch.time()
            output = function(*args, **kwargs)
            total_time = stopwatch.time() - start_time
            
            if function.__name__ == 'multiply':
                str_mult = '_float' if type(args[1]) == float else `sum([i == j for i, j in zip(args[0].shape, args[1].shape)])` + '_of_' + `args[0].ndim`
                profiler.update('multiply' + str_mult, total_time)

            elif function.__name__ == 'append':
                str_append = '_tuple' if type(args[1]) == tuple else '_micmat'
                profiler.update('append' + str_append, total_time)

            elif function.__name__ == 'sum':
                str_sum = '2' if (len(args) == 1 or args[1] == 2) else `args[1]`
                profiler.update('sum' + str_sum, total_time)

            elif function.__name__ == 'sum_replace':
                str_sum = '2' if (len(args) == 2 or args[2] == 2) else `args[2]`
                profiler.update('sum_replace' + str_sum, total_time)

            # elif function.__name__ == 'permute_dimensions':
            #     # str_permute = `args[1]`
            #     str_permute = `args[0].shape`
            #     profiler.update('permute_dimensions' + str_permute, total_time)

            # elif function.__name__ == 'convolution_gradient':
            #     str_sum = `args[-2]`
            #     profiler.update('convolution_gradient' + str_sum, total_time)

            elif function.__name__ == 'max_and_arg':
                str_max = '2' if len(args) == 1 else `args[1]`
                profiler.update('max_and_arg' + str_max, total_time)

            elif function.__name__ == 'update':
                if type(args[1]) in {float, int}:
                    str_update = '_const'
                elif args[0].shape == args[1].shape:
                    str_update = '_same'
                else:
                    str_update = `args[0].shape` + '_' + `args[1].shape`
                profiler.update('update' + str_update, total_time)

            else:
                profiler.update(function.__name__, total_time)

            return output
    return profiled_routine


stopwatch = Timer()
profiler = Profiler()
