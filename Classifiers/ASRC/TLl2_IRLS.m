function Coeff = TLl2_IRLS( X_Tst , X_Trn , lambda , display  )

% Reference: Jing Wang, Canyi Lu, et al., "Robust Face Recognition via Adaptive Sparse Representation,"
% IEEE Transactions on System, Man and Cybernetics-Part B, 2014.
% Created by Canyi Lu (canyilu@gmail.com) and Jing Wang (jw998@rutgers.edu).
% Please cite our paper if you use the code, thank you.

% Input:
% X_Tst: dim*num_trn
% X_Trn: dim*num_tsn

% Output:
% Coeff: num_trn*num_Tst
% ojbect function: test  =  train * Coeff
% dim: dim*num_test =  (dim*num_train) *(num_train * num_test)

if nargin<4
    display = false ;
end

if nargin < 3
    lambda = 0.01 ;
end

[dim,num_Trn] = size(X_Trn) ;
num_Tst = size(X_Tst,2) ;

Coeff = zeros( num_Trn , num_Tst ) ;

maxiter = 50 ; % 10, 20, 50
rc = 0.01 ;
rho = 1.4 ;
tol = 1e-8 ;
tol2 = 1e-7;
I = eye(dim,dim) ;
XtX = X_Trn'*X_Trn ;


mu = rc*norm(X_Trn,2) ;
M = eye(dim,dim) ;
N = M ;
z_old = zeros(num_Trn,1) ;
temp = X_Trn';
for i = 1 : num_Tst
    y = X_Tst(:,i) ;
    for t = 1 : maxiter
        % update z
        %         z = (lambda*diag(diag(X_Trn'*M*X_Trn))+XtX)\X_Trn'*y ;
        z = (lambda*diag(diag(temp*M*X_Trn))+XtX)\temp*y ;
        
        % update M
        %         M = (X_Trn*diag(z.^2)*X_Trn'+mu^2*I)^(-0.5) ;
        M = (X_Trn*diag(z.^2)*temp+mu^2*I)^(-0.5) ;
        % update mu
        mu = mu/rho ;
        
        % calculate obj
        if display
            obj(t) = norm(y-X_Trn*z,2)^2 + lambda * nuclearnorm(X_Trn*diag(z)) ;
        end
        
        if mu < tol || norm(z_old-z,'fro')/norm(z,'fro')<tol2
            break ;
        end
        z_old = z ;
    end
    Coeff(:,i) = z ;    
end


