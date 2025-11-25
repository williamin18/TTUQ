clear variables
load('Tests/RPC_TT_test/e3_NMOS_DA_test/NMOS_DA_data.mat')
[~,d] = size(training_samples);

xi_train = training_samples(1:3000,:);
y_train = vouts_train(1:3000,:);
m=3;

f_k = 20;

[y_predict2,n_rpc,d_rpc] = rpc_total(xi_train,y_train(:,f_k),test_samples,m,'Hermite',0.5,0.01,1e-3);
norm(y_predict2-vouts_test(:,f_k))/norm(vouts_test(:,f_k))


% [y_predict2] = pc_collocation_total(xi_train,y_train(:,f_k),test_samples,m,'Hermite');
% norm(y_predict2-vouts_test(:,f_k))/norm(vouts_test(:,f_k))


% m = 3;
% N = (m+1)*ones(d,1);
% r = 3;
% x = TTrand(N,r);
% x{1}(1,:) = [1 zeros(1,r-1)];
% for i = 1:d-1
%     x{i}(1:r,:)=eye(r);
% end
% tic
%  [y_predict2,PC_coefficients,training_err,test_err,n_iterations] = pc_collocation_tensor_optimization...
%      (xi_train,y_train(:,f_k),x,test_samples,m,'Hermite','TT-Newton',0.3,0.2,5e-3,10/11,3,true);
% toc

f = figure(5);
Hmc = histogram(abs(vouts_test(:,f_k)) ,50,'Normalization','pdf', 'DisplayStyle','bar', 'FaceColor',[0.7 0.7 0.7]);
hold on
grid on
[N_mc,edges] = histcounts(abs(y_predict2), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_mc,'b:','LineWidth',2)