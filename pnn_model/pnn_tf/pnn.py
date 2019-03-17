import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class PNN(BaseEstimator, TransformerMixin):

    def __init__(self,
                 feature_size, field_size,
                 embedding_size=8,
                 deep_layers=[32, 32], deep_init_size=50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 greater_is_better=True,use_inner=True):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result,self.valid_result = [],[]

        self.use_inner = use_inner
        self._init_graph()

    def _init_graph(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            #todo:训练数据的特征索引 [batch_size,field_size]
            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None, None],
                                             name='feat_index')

            # todo:训练数据的特征值 [batch_size,field_size]
            self.feat_value = tf.placeholder(tf.float32,
                                             shape=[None, None],
                                             name='feat_value')

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # Embeddings
            # todo: 输入的每个的特征转为通过 embedding_lookup 找到embedding层权重 [batch_size,field_size,embedding_size]
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            # todo: 特征值矩阵转为[batch_size,field_size,1]
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])

            # todo:特征值与embedding层权重相乘得到结果embedding输出值 [batch_size,field_size,embedding_size]
            self.embeddings = tf.multiply(self.embeddings,feat_value) # N * F * K

            # Linear Singal
            # todo:product层中线性部分的权重共 deep_init_size个 每一个权重的为 [field_size,embedding_size]
            linear_output = []
            for i in range(self.deep_init_size):

                single_product_linear_weight = self.weights['product-linear'][i] #todo product层中单个权重矩阵 [field_size,embedding_size]
                linear_output.append(
                    tf.reshape(
                        tf.reduce_sum(
                            tf.multiply(self.embeddings,single_product_linear_weight),axis=[1,2]), #todo 单个权重矩阵与embedding层相乘后 [batch_size,field_size,embedding_size] 相当于论文中的矩阵相乘后求和得到一个数值
                        shape=(-1,1)
                    )
                )# N * 1

            self.lz = tf.concat(linear_output, axis=1)  # N * init_deep_size

            # Quardatic Singal
            # todo:product层中内积的权重也是一共 deep_init_size个 每一个权重为 [field_size]
            quadratic_output = []
            if self.use_inner:
                for i in range(self.deep_init_size):

                    single_product_inner_weight = tf.reshape(self.weights['product-quadratic-inner'][i], (1,-1,1)) # todo product层中单个权重矩阵 [1,field_size], reshape为[1,field_size,1]

                    theta = tf.multiply(self.embeddings,single_product_inner_weight)  #todo 单个权重矩阵与embedding层相乘后 [batch_size,field_size,embedding_size]
                    quadratic_output.append(
                        tf.reshape(
                            tf.norm(tf.reduce_sum(theta, axis=1), axis=1), shape=(-1, 1) #todo 在第一维度上求和 [batch_size,embedding_size], 后对每一行求2范数 [batch_size,1] ==> 向量内积和向量二范数基本等价
                        )
                    ) # N * 1

            else:
                embedding_sum = tf.reduce_sum(self.embeddings, axis=1)
                p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))  # N * K * K
                for i in range(self.deep_init_size):
                    theta = tf.multiply(p, tf.expand_dims(self.weights['product-quadratic-outer'][i], 0))  # N * K * K
                    quadratic_output.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=(-1, 1)))  # N * 1

            self.lp = tf.concat(quadratic_output, axis=1)  # N * init_deep_size
            self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz, self.lp), self.weights['product-bias']))
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])




    def _initialize_weights(self):

        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings')

        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        # Product Layers
        if self.use_inner:
            weights['product-quadratic-inner'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.field_size], 0.0, 0.01))
        else:
            weights['product-quadratic-outer'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.embedding_size, self.embedding_size], 0.0, 0.01))


        weights['product-linear'] = tf.Variable(
            tf.random_normal([self.deep_init_size, self.field_size, self.embedding_size], 0.0, 0.01))
        weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size, ], 0, 0, 1.0))

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.deep_init_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
        )

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['output'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                                        dtype=np.float32)
        weights['output_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def predict(self, Xi, Xv,y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    # todo: Xi_train-—[训练样本数*特征索引]
    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape((-1, 1))
                loss = self.predict(Xi_valid, Xv_valid, y_valid)
                print("epoch", epoch, "loss", loss)


    def fit_on_batch(self,Xi,Xv,y):

        feed_dict = {self.feat_index:Xi,
                     self.feat_value:Xv,
                     self.label:y,
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)

        return loss

if __name__ == '__main__':
    pnn_params = {
        "embedding_size": 8,
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layer_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "verbose": True,
        "random_seed": 2017,
        "deep_init_size": 50,
        "use_inner": True,
        "feature_size":256,
        "field_size":40
    }
    pnn = PNN(**pnn_params)