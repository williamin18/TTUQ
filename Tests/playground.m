A = rand(10,5);
x_true = rand(3,1);
L = rand(5,3);
[Q,R] = qr(L,'econ');
y_true = Q*x_true;
y_true2 = y_true + 0.01*rand(5,1);

b = A*y_true;
[U,S,V] = svd(A,"econ");

V2 = V(:,1:3);
x = (A*V2)\b;
y = V2*x;

