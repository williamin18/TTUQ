clear variables
load('Tests/e2_double_tlnet/e2_data.mat')

training_samples = training_samples(1:660,:);
vouts_train = vouts_train(1:660,:);
[n_samples,d] = size(training_samples);


m = 3;
N = (m+1)*ones(d,1);
r = 3;
x = TTrand(N,r);
x{1}(1,:) = [1 zeros(1,r-1)];
for i = 1:d-1
    x{i}(1:r,:)=eye(r);
end

tic
 [vouts_TT,PC_coefficients,training_err,test_err,n_iterations] = pc_collocation_tensor_optimization...
     (training_samples,vouts_train,x,samples,m,'Hermite','TT-Newton',0.3,0.2,5e-3,10/11,3,true);
toc
norm(vouts_TT-vouts,'fro')/norm(vouts,'fro')
