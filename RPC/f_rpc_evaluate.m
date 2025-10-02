function y = f_rpc_evaluate(a,n_total,Phi,Psi)
    a_n = a(1:n_total,:);
    a_d = a(n_total+1:end,:);
    y = (Phi*a_n)./(1+Psi*a_d);
end

