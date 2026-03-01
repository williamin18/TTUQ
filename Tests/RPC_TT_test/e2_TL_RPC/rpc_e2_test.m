% clear variables
load('Tests/RPC_TT_test/e2_TL_RPC/coupled_tl_28par_10std.mat')
[~,d] = size(training_samples);

xi_train = training_samples;
y_train = vouts_train;
m=2;
f_k = 70;

[y_predict2,n_rpc,d_rpc] = rpc_total(xi_train,y_train(:,f_k),samples,m,'Hermite',0.3,0.01,5e-3);
norm(y_predict2-vouts(:,f_k))/norm(vouts(:,f_k))


[~,d] = size(xi_train);
N = (m+1)*ones(d,1);
x = TTrand(N,3);
x = TTorthogonalizeLR(x);
x = TTorthogonalizeRL(x);


[y_predict2,N_coefficients,D_coefficients,n_iterations] = rpc_TT(xi_train,y_train(:,f_k),x,samples,m,'Hermite',...
    0.3,0.1,1e-3,0.9,3);
err1 = norm(y_predict2-vouts(:,f_k))/norm(vouts(:,f_k))
% 
% 
% [y_predict3,PC_coefficients,training_err,test_err,n_iterations2] = pc_collocation_tensor_optimization...
%      (xi_train,y_train(:,f_k),x,samples,m,'Hermite','TT-Newton',0.3,0.1,1e-3,0.9,3,true);
% err2 =norm(y_predict3-vouts(:,f_k))/norm(vouts(:,f_k))


f = figure(5);
Hmc = histogram(abs(vouts(:,f_k)) ,50,'Normalization','pdf', 'DisplayStyle','bar', 'FaceColor',[0.7 0.7 0.7]);
hold on
grid on
[N_mc,edges] = histcounts(abs(y_predict2), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_mc,'b:','LineWidth',2)