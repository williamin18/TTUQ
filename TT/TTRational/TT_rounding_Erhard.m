function x = TT_rounding_Erhard(A,x,b,max_rank)
%TT_ROUNDING_ERHARD Summary of this function goes here
%   Detailed explanation goes here
x = TTorthogonalizeRL(x); 
[d,m,r] = TTsizes(x);
[~,yr] = Ax_right(A,x,1);   

for i = 1:d-1
    Ayr_i = 

end
end

