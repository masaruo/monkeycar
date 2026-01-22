import numpy as np
import logging
from tqdm import trange


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, images: np.ndarray, steerings: np.ndarray, throttles: np.ndarray, cfg)
        self.images = images
        self.steerings = steerings
        self.throttles = throttles
        self.cfg = cfg
        logger.info("Trainer Initiated")

# class Trainer:
#     def __init__(self, network, x_train, t_train, x_test, t_test, epochs=20, mini_batch_size=64, optimizer=None):
#         self.network = network
#         self.x_train = x_train
#         self.t_train = t_train
#         self.x_test = x_test
#         self.t_test = t_test
#         self.epochs = epochs
#         self.batch_size = mini_batch_size
#         self.optimizer = optimizer

#         self.train_size = x_train.shape[0]
#         self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
#         self.max_iter = int(epochs * self.iter_per_epoch)
#         self.current_iter = 0
#         self.current_epoch = 0

#         self.train_loss_list = []
#         self.test_loss_list = []

#     def train(self):
#         for epoch in trange(self.epochs):
#             idx = np.random.permutation(self.train_size)
#             x_shuffled = self.x_train[idx]
#             y_shuffled = self.y_train[idx]
#             # ミニバッチ抽出
#             for i in trange(self.iter_per_epoch):
#                 batch_x = x_shuffled[i*self.batch_size : (i + 1)*self.batch_size] #todo i maybe last? then i + 1 is out of bound
#                 batch_y = y_shuffled[i*self.batch_size : (i + 1)*self.batch_size] #todo i maybe last? then i + 1 is out of bound
            
#             # 勾配
#             grads = self.network.gradient(batch_x, batch_t)

#             # パラメーター更新
#             self.optimizer.update(self.network.params, grads)

#             loss = self.network.loss(batch_x, batch_y)
#             self.train_loss_list.append(loss)
#             self.current_iter += 1

#         # end of an epoch
#         self.current_epoch += 1
#         test_loss = self.network.loss(self.x_test, self.test_test)#? test_test?
#         self.test_loss_list.append(test_loss)


