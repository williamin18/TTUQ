function [dfdy,Cy] = Ay_denominator(Ax,C,y)
%AY_R1 undefined
%   undefined
    [n_samples,d] = size(C);


    Cy = ones(n_samples,1);
    for i = 1:d
        Cy = Cy.*(1+C(:,i)*y(i));
    end

   f = Ax./Cy;
   dfdy = zeros(n_samples,d);
   for i = 1:d
        dfdy(:,i) = -f./(1+C(:,i)*y(i)).*C(:,i);
   end
    
end