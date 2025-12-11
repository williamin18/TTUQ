function [y_predict,RPC_coefficients,n_iterations] = rpc_TT(xi_train,y_train,x,xi_test,order,polynomial,...
    preprocessing_parameter,regularization_parameter,tol,training_proportion,r_max)

max_iterations = 50;

lambda1 = preprocessing_parameter;
lambda2 = regularization_parameter;

[n_samples,d] = size(xi_train);
n_train = round(n_samples*training_proportion);
training_samples = genPolynomialSamplesTensor(xi_train(1:n_train,:),order,polynomial);
training_out = y_train(1:n_train,:);
vali_samples = genPolynomialSamplesTensor(xi_train(n_train+1:end,:),order,polynomial);
vali_out =  y_train(n_train+1:end,:);
test_samples = genPolynomialSamplesTensor(xi_test,order,polynomial);



[~,n_y] = size(y_train);
%preconditioning
for i = 1:d
    for j = 1:order
        training_samples{i}(:,j+1) = training_samples{i}(:,j+1)*lambda1^j;
        vali_samples{i}(:,j+1) = vali_samples{i}(:,j+1)*lambda1^j;
        test_samples{i}(:,j+1) = test_samples{i}(:,j+1)*lambda1^j;
    end
end

%approximate TT coefficients
RPC_coefficients = cell(n_y,1);
training_err = zeros(n_y,1);
test_err = zeros(n_y,1);
n_iterations = zeros(n_y,1);
for i = 1:n_y
    % [x,training_err(i),test_err(i),n_iterations(i)] = TT_RPC_ALS(training_samples,x,training_out(:,i),r_max,tol,max_iterations,vali_samples,vali_out,lambda2);
    [x,training_err(i),test_err(i),n_iterations(i)] = TT_Newton_rational4(training_samples,x,training_out(:,i),r_max,tol,max_iterations,vali_samples,vali_out(:,i),lambda2);
    % [x,training_err(i),test_err(i),n_iterations(i)] = TT_RPC_SGD(training_samples,x,training_out(:,i),r_max,tol,max_iterations,vali_samples,vali_out,lambda2);
    disp([training_err(i) test_err(i) n_iterations(i)])
    RPC_coefficients{i} = x;
end

[n_test_samples,~] = size(xi_test);
y_predict = zeros(n_test_samples,n_y);
for i = 1:n_y
    x = RPC_coefficients{i};
    [~,~,tt_ranks] = TTsizes(x);
    x_n = x(1:d);
    x_n{d} = x_n{d}*x{d+1}(1:tt_ranks(d+1));
    x_d = x(1:d);
    x_d{d} = x_d{d}*x{d+1}(tt_ranks(d+1)+1:2*tt_ranks(d+1));
    y_predict(:,i) = multi_r1_times_TT(test_samples,x_n)./multi_r1_times_TT(test_samples,x_d);
end

end