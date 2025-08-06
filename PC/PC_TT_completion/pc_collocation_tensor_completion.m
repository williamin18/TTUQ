function [b_predict,PC_coefficients,training_err,test_err] = pc_collocation_tensor_completion...
    (sample_indices,b_train,y_init,xi_predict,order,polynomial,method,max_rank,tol,display_outputs)
%y is TT-estimation of outputs, b is sample outputs,sample_indices are indices of b,
% xi is polynomial samples(not used on training)
switch method
    case "TT-Riemannian"
        f = @TT_Riemannian_completion;
    otherwise
        err('Unsupported method')
end

switch polynomial
    case "Hermite"
        poly = @genHermite;
    case "Legendre"
        poly = @genLegendre;
    otherwise
        err('Unsupported polynomial')
end
    
%% initialization
[n_samples,d] = size(sample_indices);


n_train = round(0.9*n_samples);
training_indices = sample_indices(1:n_train,:);
training_samples = b_train(1:n_train,:);
test_indices = sample_indices(n_train+1:end,:);
test_samples = b_train(n_train+1:end,:);


training_sample_selector = index2selector(training_indices,order+1);
test_sample_selector = index2selector(test_indices,order+1);

%% TT outs estimation
[~,n_b] = size(b_train);
TT_outs = cell(n_b,1);
training_err = zeros(n_b,1);
test_err = zeros(n_b,1);
n_iterations = zeros(n_b,1);

for i = 1:n_b
    [yi,training_err(i),test_err(i),n_iterations(i)] = f(training_sample_selector,training_samples(:,i),y_init{i},max_rank,tol,200,test_sample_selector,test_samples(:,i));
    if display_outputs 
        disp([training_err(i),test_err(i),n_iterations(i)])
    end
    TT_outs{i} = yi;
end

%% compute PC coefficients
PC_coefficients = cell(n_b,1);
polynomial_roots =  genPolynomialRoots(order+1,polynomial);
A_r1 = zeros(order+1,order+1);
for i = 1:order+1
    A_r1(:,i) = poly(polynomial_roots,i-1);
end
A_r1_inverse = A_r1^-1;
for i = 1:n_b
    PC_coefficients{i} = TT_outs{i};
    [~,m,r] = TTsizes(PC_coefficients{i});
    for j = 1:d
        Wj = PC_coefficients{i}{j};
        Wj = reshape(Wj,[r(j),m(j),r(j+1)]);
        Wj = reshape(permute(Wj, [2 1 3]),m(j),[]);
        Wj = A_r1_inverse*Wj;
        Wj = permute(reshape(Wj,m(j),r(j),r(j+1)),[2 1 3]);
        Wj = reshape(Wj,m(j)*r(j),[]);
        PC_coefficients{i}{j} = Wj;
    end
end

%% predict outputs
predict_samples = genPolynomialSamplesTensor(xi_predict,order,polynomial);
[n_predict_samples,~] = size(xi_predict);
b_predict = zeros(n_predict_samples,n_b);

for i = 1:n_b
    b_predict(:,i) = multi_r1_times_TT(predict_samples,PC_coefficients{i});
end
end

