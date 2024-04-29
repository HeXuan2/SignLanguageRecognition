
"""
这段代码中包含了两个函数，分别用于解析模型配置文件和数据服务器配置文件的内容。
parse_model_config(path): 该函数用于解析模型配置文件，首先打开文件并逐行读取内容，然后去除空行和以#开头的注释行。接着将每一行的首尾空格去除，并将非注释行按照"="进行分割，并存储到一个字典中，最终返回这个字典。
parse_data_config(path): 这个函数用于解析数据和服务器配置文件，同样是打开文件并逐行读取内容，然后去除空行和以#开头的注释行。接着将每一行按照"="进行分割，并存储到一个字典中，最终返回这个字典。
这些函数看起来是用于读取和解析配置文件的工具函数，能够方便地将配置信息提取出来并以字典的形式返回。如果你对其中任何一个函数有疑问或需要进一步解释，请随时告诉我。
"""

def parse_model_config(path):
    """Parses the model configuration file"""
    fp = open(path, 'r', encoding='utf-8')
    lines = fp.readlines()
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = dict()
    model_name = "default"
    for line in lines:
        if line.startswith('['):
            model_name = line[1:-1].rstrip()
            module_defs[model_name] = dict()
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[model_name][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data and server configuration file"""
    options = dict()
    with open(path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
    return options
