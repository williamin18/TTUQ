clear variables
close all
load('Tests/RPC_TT_test/e4_LNA_IR/rpc_e4_data2.mat')

[~,n_fpoint] = size(gain_train);
m = 2;
xi_train = training_samples;
y_train = gain_train;
y_test = gain_test;
f_k = 1:100;


[y_predict,n_rpc,d_rpc] = rpc_total(xi_train,y_train(:,f_k),test_samples,m,'Hermite',1,0,5e-3);
norm(y_predict-y_test(:,f_k),'fro')/norm(y_test(:,f_k),'fro')



