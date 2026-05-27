function [b_predict,N_coefficients,D_coefficients,n_iterations] = ...
    rpc_TT_freq2(xi_train,b_train,xi_test,freq_test,order,freq_order,polynomial,...
    preprocessing_parameter,regularization_parameter,tol,training_proportion,r_max)

max_iterations = 500;

lambda1 = preprocessing_parameter;
lambda2 = regularization_parameter;

freq = xi_train(:,end);
f_scale = 2/(max(freq)-min(freq));
f_bias = -1 - min(freq)*f_scale;
xi_train(:,end) = freq*f_scale + f_bias;

freq_test = freq_test*f_scale + f_bias;

[n_samples,d] = size(xi_train);
n_train = round(n_samples*training_proportion);
training_samples = genPolynomialFreqSamplesTensor2(xi_train(1:n_train,:),order,polynomial,freq_order);
training_out = b_train(1:n_train);
vali_samples = genPolynomialFreqSamplesTensor2(xi_train(n_train+1:end,:),order,polynomial,freq_order);
vali_out =  b_train(n_train+1:end);
test_samples = genPolynomialFreqSamplesTensor(xi_test,order,polynomial,freq_test,freq_order);



[~,n_y] = size(b_train);
%preconditioning
for i = 1:d-1
    for j = 1:order
        training_samples{i}(:,j+1) = training_samples{i}(:,j+1)*lambda1^j;
        vali_samples{i}(:,j+1) = vali_samples{i}(:,j+1)*lambda1^j;
        test_samples{i}(:,j+1) = test_samples{i}(:,j+1)*lambda1^j;
    end
end

%init TT coefficients
N = [(order+1)*ones(d-1,1); freq_order+1];
% r1_init = cell(d,1);
% for i = 1:d-1
%     r1_init{i} = zeros(order+1,1);
%     r1_init{i}(1) = 1;
%     r1_init{i}(2) = 0.1;
% end
% r1_init{d} = zeros(freq_order+1,1);
% r1_init{d}(1) = mean(b_train(1,:));
% r1_init{d}(2) = std(b_train(1,:));
% 
% 
% N_coefficients = TTaxby(1,r1_init,1,TTrand(N,r_max-1));
N_coefficients = TTrand(N,r_max);
D_coefficients = TTrand(N,r_max);

[N_coefficients,D_coefficients,training_err,test_err,n_iterations] = TT_Newton_rational_r1d3(...
    training_samples,N_coefficients,training_samples,D_coefficients,training_out,...
    r_max,tol,max_iterations,vali_samples,vali_samples,vali_out,lambda2);


[n_test_samples,~] = size(xi_test);
b_predict = multi_r1_times_TT(test_samples,N_coefficients)./(1+multi_r1_times_TT(test_samples,D_coefficients));
b_predict = reshape(b_predict,n_test_samples,[]);

% b_predict = zeros(n_test_samples,n_y);
% C_test = zeros(n_test_samples,d);
% for i = 1:d
%     C_test(:,i) = test_samples{i}(:,2);
% end
% for i = 1:n_y
%     x = N_coefficients{i};
%     y = D_coefficients{i};
%     b_predict(:,i) = multi_r1_times_TT(test_samples,x)./(1+multi_r1_times_TT(test_samples,y));
%     % b_predict(:,i) = multi_r1_times_TT(test_samples,x)./Cy_r1(C_test,y);
%     % b_predict(:,i) = multi_r1_times_TT(test_samples,x)./(1+C_test*y);
% end

end