function [b_predict,N_coefficients,D_coefficients,n_iterations] = ...
    rpc_TT(xi_train,b_train,x,xi_test,order,polynomial,...
    preprocessing_parameter,regularization_parameter,tol,training_proportion,r_max)

max_iterations = 200;

lambda1 = preprocessing_parameter;
lambda2 = regularization_parameter;

[n_samples,d] = size(xi_train);
n_train = round(n_samples*training_proportion);
training_samples = genPolynomialSamplesTensor(xi_train(1:n_train,:),order,polynomial);
training_out = b_train(1:n_train,:);
vali_samples = genPolynomialSamplesTensor(xi_train(n_train+1:end,:),order,polynomial);
vali_out =  b_train(n_train+1:end,:);
test_samples = genPolynomialSamplesTensor(xi_test,order,polynomial);



[~,n_y] = size(b_train);
%preconditioning
for i = 1:d
    for j = 1:order
        training_samples{i}(:,j+1) = training_samples{i}(:,j+1)*lambda1^j;
        vali_samples{i}(:,j+1) = vali_samples{i}(:,j+1)*lambda1^j;
        test_samples{i}(:,j+1) = test_samples{i}(:,j+1)*lambda1^j;
    end
end

%approximate TT coefficients
N_coefficients = cell(n_y,1);
D_coefficients = cell(n_y,1);
training_err = zeros(n_y,1);
test_err = zeros(n_y,1);
n_iterations = zeros(n_y,1);
y = zeros(d,1);
for i = 1:n_y
    [x,y,training_err(i),test_err(i),n_iterations(i)] = TT_Newton_rational7(training_samples,x,training_out(:,i),y,r_max,tol,max_iterations,vali_samples,vali_out(:,i),lambda2);
    % [x,training_err(i),test_err(i),n_iterations(i)] = TT_Newton_rational4(training_samples,x,training_out(:,i),r_max,tol,max_iterations,vali_samples,vali_out(:,i),lambda2);
    disp([training_err(i) test_err(i) n_iterations(i)])
    N_coefficients{i} = x;
    D_coefficients{i} = y;
end

[n_test_samples,~] = size(xi_test);
b_predict = zeros(n_test_samples,n_y);
C_test = zeros(n_test_samples,d);
for i = 1:d
    C_test(:,i) = test_samples{i}(:,2);
end
for i = 1:n_y
    x = N_coefficients{i};
    y = D_coefficients{i};
    b_predict(:,i) = multi_r1_times_TT(test_samples,x)./(1+C_test*y);
end

end