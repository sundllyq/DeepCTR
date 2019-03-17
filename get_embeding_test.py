import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Embedding, Dense, Reshape, Concatenate
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2


sparse_features=['C1','C2','C3']
dense_features=['I1','I2']

res:DataFrame = pd.read_csv("./criteo_sample.txt").loc[5:25,sparse_features+dense_features]
res:DataFrame = res.reset_index(drop=True)

res[sparse_features]=res[sparse_features].fillna('-1',)
res[dense_features]=res[dense_features].fillna(0,)

# 稀疏特征转为数字标签
for feat in sparse_features:
    lbe = LabelEncoder()
    res[feat]=lbe.fit_transform(res[feat])

# 稠密特征归一化
mns = MinMaxScaler(feature_range=(0,1))
res[dense_features]=mns.fit_transform(res[dense_features])


sparse_feature_dict={feat:res[feat].nunique() for feat in sparse_features}
dense_feature_list=dense_features

print(sparse_feature_dict)
print(dense_feature_list)

feature_dim_dict = {"sparse":sparse_feature_dict,"dense":dense_feature_list}

#todo #得到输入的embedding

#todo #1、创建输入层tensor
sparse_input_dict = {feat:Input(shape=(1,),name="sparse_"+str(i)+'-'+feat) for i,feat in enumerate(feature_dim_dict["sparse"])}
dense_input_dict = {feat:Input(shape=(1,),name="dense_"+str(i)+'-'+feat) for i,feat in enumerate(feature_dim_dict["dense"])}

#todo #2、创建输入层的中稀疏特征的embedding
embedding_size=8
l2_reg=1e-5
init_std=0.0001
seed=1024
deep_sparse_emb_dict = \
    {
        feat:
            Embedding(
                feature_dim_dict["sparse"][feat],
                embedding_size,
                embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                embeddings_regularizer=l2(l2_reg),
                name='sparse_emb_' + str(i) + '-' + feat
            )
            for i, feat in enumerate(feature_dim_dict["sparse"])
    }
#todo #3、将稀疏输入tensor 通过embedding层 得到结果
deep_emb_list = [deep_sparse_emb_dict[feat](v) for feat,v in sparse_input_dict.items()]

#todo #4、合并 稠密向量的tensor reshape 成 embedding 和 稀疏特征的 embedding tensor 组成一个list


continuous_embedding_list = map(Dense(embedding_size,use_bias=False,kernel_regularizer=l2(l2_reg),), list(dense_input_dict.values()))
continuous_embedding_list = list(map(Reshape((1, embedding_size)), continuous_embedding_list))

deep_emb_list+=continuous_embedding_list

