clear variables
close all
load('Tests/RPC_TT_test/e4_LNA_IR/rpc_e4_data2.mat')

[~,n_fpoint] = size(gain_train);
m = 2;
xi_train = training_samples;
y_train = gain_train;
y_test = gain_test;
f_k = 1:100;


tic
[y_predict,n_rpc,d_rpc] = rpc_total(xi_train,y_train(:,f_k),test_samples,m,'Hermite',1,0.01,5e-3);
toc
norm(y_predict-y_test(:,f_k),'fro')/norm(y_test(:,f_k),'fro')




y_mean0 = mean(abs(y_test));
y_sigma0 = std((y_test));

y_mean1 = mean(abs(y_predict));
y_sigma1 = std((y_predict));


fpoints = linspace(1e9,3e9,100);
scale = 1e9;
f = figure(1);
subplot(2,1,1);
hold on
plot(fpoints/scale,y_mean0,'k-','LineWidth',3);
plot(fpoints/scale,y_mean1,'g--','LineWidth',2);
legend('Monte Carlo simulation','Totol order RPC')
grid on
xlabel('Frequency (GHz)','interpreter','LaTex')
ylabel('Mean of Gain','interpreter','LaTex')
set(gca,'GridLineStyle','--')
set(gca, 'FontName', 'Times New Roman')
set(gca,'FontSize',12)
box on;

subplot(2,1,2);
hold on
plot(fpoints/scale,y_sigma0,'k-','LineWidth',3);
plot(fpoints/scale,y_sigma1,'g--','LineWidth',2);
grid on
ylabel('Standard devation of of Gain','interpreter','LaTex')
xlabel('Frequency (GHz)','interpreter','LaTex')
set(gca,'GridLineStyle','--')
set(gca, 'FontName', 'Times New Roman')
set(gca,'FontSize',12)
box on;

f.Position = [100 100 675 500];