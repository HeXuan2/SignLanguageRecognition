import logging
import os

"""
这段代码定义了一个Logger类，用于记录日志。该类的功能是创建一个日志文件并设置相应的logging配置。下面是对这段代码的一些解释：

如果指定的log_path目录不存在，就会创建一个'./log'目录，并在该目录下创建一个空文件，文件名为log_path。

设置logging的级别为INFO级别，并定义了输出的格式，包括时间、文件名、行号、日志级别和消息。

使用FileHandler和StreamHandler分别输出到文件和屏幕，两者的级别也都是INFO级别。

最后通过addHandler方法将两个Handler添加到logger中，这样就可以同时将日志输出到文件和屏幕。

需要注意的是，如果在多个地方创建Logger实例，可能会导致日志重复输出到文件中。因此，在使用Logger类时需要特别注意控制日志的输出。
"""

class Logger(object):
    def __init__(self, log_path):

        if not os.path.exists(log_path):
            # os.mkdir('./log')
            with open(log_path, "w") as f:
                pass
        # 设置logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # 使用FileHandler输出到文件
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # 添加两个Handler
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

