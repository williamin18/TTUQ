function [r] = genPolynomialRoots(order,polynomial)
switch polynomial
    case "Hermite"
        switch order
            case 1
                p = [1 0];
            case 2
                p = [1 0 -1];
            case 3
                p = [1 0 -3 0];
            case 4
                p = [1 0 -6 0 3];
            case 5
                p = [1 0 -10 0 15 0];
            case 6
                p = [1 0 -15 0 45 0 -15];
            otherwise
                err('Unsupported order')
        end
    case "Legendre"
        switch order
            case 1
                p = [1 0];
            case 2
                p = 0.5*[3 0 -1];
            case 3
                p = 0.5*[5 0 -3 0];
            case 4
                p = 1/8*[35 0 -30 0 3];
            case 5
                p = 1/8*[63 0 -70 0 15 0];
            case 6
                p = 1/16*[231 0 -315 0 105 0 -5];
            otherwise
                err('Unsupported order')
        end
    otherwise
        err('Unsupported polynomial type')
end

r = sort(roots(p));
end

