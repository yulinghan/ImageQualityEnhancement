function [fout] = multipatchpacking(f,S1, S2, K, h, box)
[~, N] = size(f);
fout = zeros(1, N);
L = (4*S2 + 1);
l_NN =  S1*N + N + 0.5*S1*(2*K-S1-1);
L_tot = l_NN + (N-S1) + S1*K + N*(S1-1) - 0.5 * S1 * (S1-1);
F = zeros(L_tot, L);
F_bar = zeros(L_tot, L);
l = 0;
for mu = 1 : (S1 + 1)
    for nu =  1  : (N - (S1 + 1) + mu)
        l = l + 1;
        for lp = 1 :(2* S2 +1)
            F (l, lp) =  f(lp, nu)' * f(lp, (S1 + 1) - mu + nu);
        end
        for lp = 1:S2
            F (l, lp + (2* S2 +1)) =  f(S2+1, (S1 + 1) - mu + nu )' * f(lp, nu);
        end
        for lp = 2 + S2 : 2 * S2 + 1
            F (l, lp + (2* S2)) =  f(S2+1, (S1 + 1) - mu + nu )' * f(lp, nu);
        end
    end
    l = l + K;
end
for mu = (S1+2) : (2*S1 + 1)
    for nu =  1  : (N - mu + S1 +1)
        l = l + 1;
        for lp = 1:S2
            F (l, lp + (2* S2 +1)) =  f(S2+1, nu )' * f(lp, nu + mu -S1 - 1);
        end
        for lp = 2 + S2 : 2 * S2 + 1
            F (l, lp + (2* S2)) =  f(S2+1, nu )' * f(lp, nu + mu -S1 - 1);
        end
    end
    l = l + K;
end
for loop = 1:L
    F_bar(:, loop) = conv(F(:, loop), box, 'same');
end
for i =  1 : N
    num1 = 0;
    denom = 0;
    l_ii = S1*N + i + 0.5*S1*(2*K-S1-1);
    for j= max(1, (i-S1)):i
        for loop1 = 1:S2
            w = exp(-(F_bar(l_ii, S2 +1)  + F_bar((S1*N + j + 0.5*S1*(2*K-S1-1)), loop1) ...
                - 2*F_bar( (S1-i+j)*(N-0.5*(S1+1+i-j)+K)+j, loop1 + (2*S2 +1) ))/(h^2) );
            num1 = num1 + w*f(loop1, j);
            denom =denom + w;
        end
        w = (exp(-( F_bar(l_ii, S2 +1) + F_bar(S1*N + j + 0.5*S1*(2*K-S1-1), S2 +1)-2*F_bar( (S1-i+j)*(N-0.5*(S1+1+i-j)+K)+j, S2 +1) )/(h^2)));
        num1 = num1 + w*f(S2 +1, j);
        denom =denom + w;
        for loop1 = S2 + 2 : (2*S2 +1)
            t1 = F_bar(l_ii, S2+1);
            t2 = F_bar((S1*N + j + 0.5*S1*(2*K-S1-1)), loop1);
            t3 = 2*F_bar( (S1-i+j)*(N-0.5*(S1+1+i-j)+K)+j, loop1 + (2*S2) );
            w = exp(-(t1  + t2 - t3 )/(h^2) );
            num1 = num1 + w*f(loop1, j);
            denom =denom + w;
        end
    end
    for j= i+1 : min(N, (i+S1))
        d = (j-i);
        l_ij = l_NN + i + d*K + N*(d-1) -0.5*d*(d-1);
        for loop1 = 1:S2
            t1 = F_bar(l_ii, S2 +1);
            t2 = F_bar((S1*N + j + 0.5*S1*(2*K-S1-1)), loop1);
            t3 = 2*F_bar(l_ij, loop1 + (2*S2 +1) );
            w = exp(-(t1  + t2 - t3 )/(h^2) );
            num1 = num1 + w*f(loop1, j);
            denom =denom + w  ;
        end
        w = (exp(-( F_bar(l_ii, S2 +1) + F_bar(S1*N + j + 0.5*S1*(2*K-S1-1), S2 +1)-2*F_bar( (S1-j+i)*(N-0.5*(S1+1+j-i)+K)+i, S2 +1) )/(h^2)));
        num1 = num1 + w*f(S2 +1, j);
        denom = denom + w;
        for loop1 = S2 + 2 : (2*S2 +1)
            t1 = F_bar(l_ii, S2 +1);
            t2 = F_bar((S1*N + j + 0.5*S1*(2*K-S1-1)), loop1);
            t3 = 2*F_bar( l_ij, loop1 + (2*S2) );
            w = exp(-(t1  + t2 - t3)/(h^2) );
            num1 = num1 + w*f(loop1, j);
            denom =denom + w;
        end
    end
    fout(i)= num1/denom;
end