clear variables
load('Tests/RPC_TT_test/e1_LPF/rpc_e1_samples.mat')

[~,n_fpoint] = size(vouts_train2);
m = 3;
x_train = training_samples2(1:3000,:);
y_train = vouts_train2(1:3000,:);
y_predict = rpc_total(x_train,y_train(:,1:50),test_samples,3,'Hermite');

norm(y_predict-vouts_test(:,1:50))/norm(vouts_test(:,1:50))