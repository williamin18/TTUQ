function [a_n,a_d,n_iterations,rel_err] = sk_solve(Phi,y,tol,max_iterations,regularizaion_parameter)

rel_err = 1;
break_counter = 0;
break_limit = 5;

[n_samples,n_n] = size(Phi);
gamma = ones(n_samples,1);
lambda = regularizaion_parameter;

for i = 1:max_iterations
    if lambda ~= 0
        b = [zeros(n_samples,1);100 ; zeros(2*n_n,1)];
        A = [Phi.*gamma, -Phi.*y.*gamma; [zeros(1,n_n) 100  zeros(1,n_n-1)]; lambda*eye(2*n_n)];
    else
        b = [zeros(n_samples,1);100];    
        A = [Phi.*gamma, -Phi.*y.*gamma; [zeros(1,n_n) 100  zeros(1,n_n-1)]];
    end
    x = A\b;
    
    a_n = x(1:n_n,:);
    a_d = x(n_n+1:end,:);
    
    rel_err_old = rel_err;
    rel_err = norm((Phi*a_n)./(Phi*a_d)-y)/norm(y);
    
    
    if rel_err<tol 
        break
    end

    if  rel_err_old-rel_err < tol/10
        break_counter = break_counter+1;
        if break_counter >break_limit
            break
        end
    else
        break_counter = 0;
    end

    gamma = 1./(Phi*a_d);
end
n_iterations = i;

end

