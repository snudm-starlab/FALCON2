# A simple torch style logger
# (C) Wei YANG 2017
'''
The following codes are from https://github.com/d-li14/mobilenetv2.pytorch
'''

from __future__ import absolute_import
from matplotlib import pyplot  as plt
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    """
    save figure function
    
    :param fname: file name
    :param dpi: set dpi of figure
    """
    dpi = 150 if dpi is None else dpi
    plt.savefig(fname, dpi=dpi)

def plot_overlap(logger, names=None):
    """
    draw plot
    
    :param logger: logger
    :param names: customized names of log
    :return: title list: list of logger titles
    """
    names = logger.names if names is None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x_data = np.arange(len(numbers[name]))
        plt.plot(x_data, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger:
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        '''
        initialize the Logger class

        :param fpath: file path
        :param title: title of logger
        :param resume: whether resume or not
        '''

        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r', encoding="utf-8")
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                print(self.names)
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i, number in enumerate(numbers):
                        self.numbers[self.names[i]].append(number)
                self.file.close()
                self.file = open(fpath, 'a', encoding="utf-8")
            else:
                self.file = open(fpath, 'w', encoding="utf-8")

    def set_names(self, names):
        """
        name setting function
        
        :param names: set names of logger
        """
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        """
        append function
        
        :param numbers: result with respect to names
        """
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write(f"{num:.6f}")
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        """
        draw plot function
        
        :param names: names of logger
        """
        names = self.names if names is None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x_data = np.arange(len(numbers[name]))
            plt.plot(x_data, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        """
        file close function
        """
        if self.file is not None:
            self.file.close()

class LoggerMonitor:
    '''Load and visualize multiple logs.'''
    def __init__ (self, file_paths):
        '''
        paths is a distionary with {name:filepath} pair
        
        param paths: file paths
        '''
        self.loggers = []
        for title, path in file_paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        """
        draw plot function
        
        :param names: names of logger
        """
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)

if __name__ == '__main__':
    # Example: logger monitor
    paths = {
    'resadvnet20':'/home/name/code/checkpoint/cifar10/resadvnet20/log.txt',
    'resadvnet32':'/home/name/code/checkpoint/cifar10/resadvnet32/log.txt'
    }

    field = ['Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')
