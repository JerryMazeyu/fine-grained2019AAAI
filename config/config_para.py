class Config(object):
    def __init__(self):
        # 训练目标网络的参数
        self.batch_size = 16
        self.num_workers = 4
        self.epochs = 25
        self.log_path = 'log.txt'
        self.num_classes = 10



opt = Config()