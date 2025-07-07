function [b_predict,PC_coefficients,training_err,test_err,n_iterations] = ...
    pc_collocation_tensor_optimization(xi_train,b_train,x,xi_predict,order,polynomial,method,...
    left_preconditioning_parameter,regularization_parameter,tol,training_proportion,r_max,print_output,varargin)
switch method
    case "TT-ALS"
        f = @TT_ALS;
        max_iteration = 1000;
    case "TT-Newton"
        f = @TT_Newton_GD;
        max_iteration = 200;
    case "TT-SGD"
        x0 = x;
        if isempty(varargin)
            batch_size = 60;
            max_iteration = 10;
        else
            batch_size = varargin{1};
            max_iteration = varargin{2};
        end
        f = @(A,b,x,r_round,tol,max_epoches,A_test,b_test,lambda)TT_SGD_linear(A,b,x,r_round,tol,max_epoches,A_test,b_test,lambda,batch_size);
    otherwise
        error('Unsupported optimization method name')
end
    


lambda1 = left_preconditioning_parameter;
lambda2 = regularization_parameter;

%train test split
[n_samples,d] = size(xi_train);
n_train = round(n_samples*training_proportion);
training_samples = genPolynomialSamplesTensor(xi_train(1:n_train,:),order,polynomial);
training_out = b_train(1:n_train,:);
test_samples = genPolynomialSamplesTensor(xi_train(n_train+1:end,:),order,polynomial);
test_out =  b_train(n_train+1:end,:);
predict_samples = genPolynomialSamplesTensor(xi_predict,order,polynomial);


[~,n_b] = size(b_train);
%preconditioning
for i = 1:d
    for j = 1:order
        training_samples{i}(:,j+1) = training_samples{i}(:,j+1)*lambda1^j;
        test_samples{i}(:,j+1) = test_samples{i}(:,j+1)*lambda1^j;
        predict_samples{i}(:,j+1) = predict_samples{i}(:,j+1)*lambda1^j;
    end
end


%approximate TT coefficients
PC_coefficients = cell(n_b,1);
training_err = zeros(n_b,1);
test_err = zeros(n_b,1);
n_iterations = zeros(n_b,1);
for i = 1:n_b
    [x,training_err(i),test_err(i),n_iterations(i)] = f(training_samples,training_out(:,i),x,r_max,tol,max_iteration,test_samples,test_out(:,i),lambda2);
    if print_output
        disp([training_err(i) test_err(i) n_iterations(i)])
    end
    PC_coefficients{i} = x;
    if method == "TT-SGD"
        x = x0;
    end
end


%predict outputs
[n_predict_samples,~] = size(xi_predict);
b_predict = zeros(n_predict_samples,n_b);
for i = 1:n_b
    b_predict(:,i) = multi_r1_times_TT(predict_samples,PC_coefficients{i});
end
end

