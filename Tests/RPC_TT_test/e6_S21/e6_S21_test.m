clear variables
close all

load('Tests\RPC_TT_test\e6_S21\S21_2par.mat');

S21_train = permute(S21_full,[2 3 1]);
S21_train = reshape(S21_train,49,[]);
xi_1 = linspace(-1,1,7)';
[i1,i2] = ind2sub([7 7], (1:49)');
xi_train = [xi_1(i1) xi_1(i2)];

S21_predict = pc_collocation_total(xi_train,S21_train,xi_test,6 ,'Legendre');
norm(S21_predict-S21_test,'fro')/norm(S21_test,'fro')

% S21_predict = pc_collocation_total(xi_train,S21_train,[-1,-1],4 ,'Legendre');
% figure(1)
% hold on
% plot(1:200,abs(S21_predict))
% plot(1:200,abs(S21_train(1,:)))