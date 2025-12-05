clear variables
load('Tests/RPC_TT_test/e2_TL_RPC/coupled_tl_28par_10std.mat')
[~,d] = size(training_samples);

xi_train = training_samples;
y_train = vouts_train;
m=2;
f_k = 70;

[y_predict2,n_rpc,d_rpc] = rpc_total(xi_train,y_train(:,f_k),samples,m,'Hermite',0.3,0.01,5e-3);
norm(y_predict2-vouts(:,f_k))/norm(vouts(:,f_k))

% 
% [~,d] = size(xi_train);
% N = [(m+1)*ones(d,1); 2];
% x = TTrand(N,3);
% x = TTorthogonalizeLR(x);
% x{d+1} = x{d+1}/norm( x{d+1},'fro');
% x = TTorthogonalizeRL(x);
% 
% 
% [y_predict2,RPC_coefficients,n_iterations] = rpc_TT(xi_train,y_train(:,f_k),x,samples,m,'Hermite',...
%     0.3,0.01,1e-3,0.9,5);
% norm(y_predict2-vouts(:,f_k))/norm(vouts(:,f_k))

f = figure(5);
Hmc = histogram(abs(vouts(:,f_k)) ,50,'Normalization','pdf', 'DisplayStyle','bar', 'FaceColor',[0.7 0.7 0.7]);
hold on
grid on
[N_mc,edges] = histcounts(abs(y_predict2), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_mc,'b:','LineWidth',2)