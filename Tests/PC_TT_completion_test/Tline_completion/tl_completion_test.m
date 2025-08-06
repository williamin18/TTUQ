clear variables
load('Tests/PC_TT_completion_test/Tline_completion/tl_completion_samples.mat')

training_idx = training_idx(1:700,:);
training_xi = training_xi(1:700,:);
training_outs = training_outs(1:700,:);

[~,d] = size(training_xi);
[n_samples,n_outs] = size(training_outs);
m = 3;
N = (m+1)*ones(d,1);
r = 3;

y_init = cell(n_outs,1);
for i = 1:n_outs
    y_init{i} = formRank1Tensor(ref_out(i),r1_outs(:,i),m,d);
end

tic
[vouts_predicted,PC_coefficients,training_err,test_err] = pc_collocation_tensor_completion...
    (training_idx,training_outs,y_init,testing_xi,m,'Hermite','TT-Riemannian',3,5e-3,true);
toc
disp(norm(vouts_predicted-testing_outs,'fro')/norm(testing_outs,'fro'))

% training_xi2 = training_xi2(1:9920,:);
% training_outs2 = training_outs2(1:9920,:);
% tic
% vouts_total = pc_collocation_total(training_xi2,training_outs2,testing_xi,m ,'Hermite');
% toc
% disp(norm(vouts_total-testing_outs,"fro")/norm(testing_outs,"fro"))
% 
% load('Tests/Tline_completion/ttcross_outs.mat')
% disp(norm((vouts_tt_cross)-testing_outs,"fro")/norm(testing_outs,"fro"))