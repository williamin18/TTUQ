function [a_n,a_d,n_iterations,rel_err] = sk_solve3(Phi,Psi,y,tol,max_iterations)

rel_err = 1;
break_counter = 0;
break_limit = 5;

[n_samples,n_n] = size(Phi);
[n_samples,n_d] = size(Psi);
lambda = ones(n_samples,1);


for i = 1:max_iterations

    b = lambda.*y;
    A = [repmat(lambda,1,n_n).*Phi, -repmat(b,1,n_d).*Psi];
    x = A\b;
    
    a_n = x(1:n_n,:);
    a_d = x(n_n+1:end,:);
    
    rel_err_old = rel_err;
    rel_err = norm((Phi*a_n)./(1+Psi*a_d)-y)/norm(y);
    
    
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

    lambda = 1./(1+Psi*a_d);
end
n_iterations = i;

end

