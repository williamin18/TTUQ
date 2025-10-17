clear variables
load('Tests/RPC_TT_test/e1_LPF/rpc_e1_samples.mat')

[~,n_fpoint] = size(vouts_train2);
m = 3;
x_train = training_samples2(1:3000,:);
y_train = vouts_train2(1:3000,:);
y_predict = rpc_total(x_train,y_train(:,100),test_samples,3,'Hermite');

norm(y_predict-vouts_test(:,100))/norm(vouts_test(:,100))

f = figure(5);
Hmc = histogram(abs(vouts_test(:,100)) ,50,'Normalization','pdf', 'DisplayStyle','bar', 'FaceColor',[0.7 0.7 0.7]);
hold on
grid on
[N_mc,edges] = histcounts(abs(y_predict), 'Normalization', 'pdf');
plot(edges(1:end-1)+(edges(2)-edges(1))/2,N_mc,'b:','LineWidth',2)