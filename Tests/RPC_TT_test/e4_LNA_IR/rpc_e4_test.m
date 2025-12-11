clear variables
close all
load('Tests/RPC_TT_test/e4_LNA_IR/rpc_e4_data.mat')

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


m = 1;
[~,d] = size(xi_train);
N = [(m+1)*ones(d,1); 2];
x = TTrand(N,3);
x = TTorthogonalizeLR(x);
x{d+1} = x{d+1}/norm( x{d+1},'fro');
x = TTorthogonalizeRL(x);

tic
[y_predict2,RPC_coefficients,n_iterations] = rpc_TT(xi_train,y_train,x,test_samples,m,'Hermite',...
    0.3,0.01,5e-3,0.9,8);
toc
norm(y_predict2-y_test)/norm(y_test)


y_mean0 = mean(abs(y_test));
y_sigma0 = std((y_test));

y_mean1 = mean(abs(y_predict));
y_sigma1 = std((y_predict));

y_mean2 = mean(abs(y_predict2));
y_sigma2 = std((y_predict2));

fpoints = linspace(10,3e6,100);
scale = 1e6;
f = figure(1);
subplot(2,1,1);
hold on
plot(fpoints/scale,y_mean0,'k-','LineWidth',3);
plot(fpoints/scale,y_mean1,'g--','LineWidth',2);
plot(fpoints/scale,y_mean2,'b--','LineWidth',1.5);
legend('Monte Carlo simulation','Totol order RPC', 'TT-RPC')
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
plot(fpoints/scale,y_sigma2,'b--','LineWidth',1.5);

grid on
ylabel('Standard devation of of Gain','interpreter','LaTex')
xlabel('Frequency (GHz)','interpreter','LaTex')
set(gca,'GridLineStyle','--')
set(gca, 'FontName', 'Times New Roman')
set(gca,'FontSize',12)
box on;

f.Position = [100 100 675 500];


f_k = 50;

f = figure(2);
Hmc = histogram(abs(y_test(:,f_k)) ,50,'Normalization','pdf', 'DisplayStyle','bar', 'FaceColor',[0.7 0.7 0.7]);
hold on
grid on
[N_k,edges] = histcounts(abs(y_predict(:,f_k)), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_k,'g:','LineWidth',2)
[N_k,edges] = histcounts(abs(y_predict2(:,f_k)), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_k,'b:','LineWidth',2)