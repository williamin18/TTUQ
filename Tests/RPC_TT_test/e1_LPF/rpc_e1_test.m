clear variables
load('Tests/RPC_TT_test/e1_LPF/rpc_e1_samples.mat')

[~,n_fpoint] = size(vouts_train2);
m = 1;
xi_train = training_samples1(1:1000,:);
y_train = vouts_train1(1:1000,:);

[y_predict2,n_rpc,d_rpc] = rpc_total(xi_train,y_train(:,100),test_samples,3,'Hermite',1,0,5e-3);
norm(y_predict2-vouts_test(:,100))/norm(vouts_test(:,100))

% [~,d] = size(xi_train);
% N = [(m+1)*ones(d,1); 2];
% x = TTrand(N,3);
% x = TTorthogonalizeLR(x);
% x{d+1} = x{d+1}/norm( x{d+1},'fro');
% x = TTorthogonalizeRL(x);
% 
% 
% [y_predict2,RPC_coefficients,n_iterations] = rpc_TT(xi_train,y_train(:,100),x,test_samples,m,'Hermite',...
%     0.3,0.01,1e-3,0.9,8);

norm(y_predict2-vouts_test(:,100))/norm(vouts_test(:,100))

f = figure(5);
Hmc = histogram(abs(vouts_test(:,100)) ,50,'Normalization','pdf', 'DisplayStyle','bar', 'FaceColor',[0.7 0.7 0.7]);
hold on
grid on
[N_mc,edges] = histcounts(abs(y_predict2), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_mc,'b:','LineWidth',2)