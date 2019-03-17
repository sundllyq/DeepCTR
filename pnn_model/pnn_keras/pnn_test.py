import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import pnn
from sklearn.metrics import log_loss, roc_auc_score
data = pd.read_csv("./criteo_sample.txt")


sparse_features  = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I'+str(i) for i in range(1,14)]
target = ['label']

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)


for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat]=lbe.fit_transform(data[feat])


mns = MinMaxScaler(feature_range=(0,1))
data[dense_features]=mns.fit_transform(data[dense_features])


sparse_feature_dict = {feat: data[feat].nunique() for feat in sparse_features}
dense_feature_list = dense_features



train, test = train_test_split(data, test_size=0.2)
train_model_input = [train[feat].values for feat in sparse_feature_dict] + \
    [train[feat].values for feat in dense_feature_list]
test_model_input = [test[feat].values for feat in sparse_feature_dict] + \
    [test[feat].values for feat in dense_feature_list]



model_input = [data[feat].values for feat in sparse_feature_dict] + [data[feat].values for feat in dense_feature_list]

feature_dim_dict = {"sparse":sparse_feature_dict,"dense":dense_feature_list}

model = pnn.PNN(feature_dim_dict=feature_dim_dict,
                embedding_size=8,
                hidden_size=(128,128),
                activation="relu",
                final_activation="sigmoid",
                use_inner=True,
                use_outter=False)

model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=4, verbose=2, validation_split=0.2,)


pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))