function Coeff = TLl2_adm( X_Tst , X_Trn , lambda , display  )

% Reference: Wang, Jing and Lu, Canyi and Wang, Meng and Li, Peipei and Yan, Shuicheng and Hu, Xuegang, "Robust Face Recognition via Adaptive Sparse Representation,"
% IEEE Transactions on System, Man and Cybernetics-Part B, 2014.
% Please cite our paper if you use the code, thank you.
% Objective function
%      min 1/2 * ||y-Xw||_2^2 + lambda * ||X*Diag(w)||_{\ast}

% Input:
% X_Tst: dim*num_trn
% X_Trn: dim*num_tsn

% Output:
% Coeff: num_trn*num_Tst
% ojbect function: test  =  train * Coeff
% dim: dim*num_test =  (dim*num_train) *(num_train * num_test)

% if nargin<2
%     norm_x = norm(X,2) ;
%     lambda = 1/(sqrt(n)*norm_x) ;
% end

if nargin<4
    display = false ;
end

if nargin < 3
    lambda = 0.001 ;
end
[dim,num_Trn] = size(X_Trn) ;
num_Tst = size(X_Tst,2) ;

Coeff = zeros( num_Trn , num_Tst ) ;

tol = 1e-8 ;
maxIter = 1000 ; %1e6  ;% ;500
% rho = 1.1 ;
tol2 = 3e-4 ;
rho0 = 1.1 ;
rho = 1.5 ;%1.1,1.2,1.3Ô½´óÔ½¿ì
max_mu = 1e10 ; 
% display = true ;

XtX = X_Trn'*X_Trn ;
diagXtX = diag(diag(XtX)) ;
XtY = X_Trn'*X_Tst ;
for i = 1 : num_Tst
    y = X_Tst(:,i) ;
    
    %% Initializing optimization variables
    mu = 1e-2 ;% tuning parameter
    w = zeros(num_Trn,1) ;
%     w = inv(riX'*riX+lambda*eye(num-1,num-1)) * riX' * y ;%alternative choice
    Z = zeros(dim,num_Trn) ;
    Y = zeros(dim,num_Trn) ;
    iter = 0 ;    
    while iter<maxIter
        iter = iter + 1; 
        w_old = w ;
        Z_old = Z ;
        
        
        %update Z
        temp = X_Trn*diag(w) - Y/mu ;
        [U,sigma,V] = svd(temp,'econ');
        sigma = diag(sigma);
        svp = length( find( sigma>lambda/mu ) ) ;
        if svp>=1
            sigma = sigma(1:svp)-lambda/mu ;
        else
            svp = 1 ;
            sigma = 0 ;
        end
        Z = U(:,1:svp)*diag(sigma)*V(:,1:svp)' ;
          
        
        %udpate w        
        A = XtX + mu*diagXtX ;
        b = XtY(:,i) + diag(X_Trn'*(Y+mu*Z)) ;
        w = A\b ;
        
        
        ymxw = y - X_Trn*w ;
        leq = Z - X_Trn*diag(w) ;        
        stopC = max(max(abs(leq))) ;
        
        if display && (iter==1 || mod(iter,1)==0 || stopC<tol)
            err = norm(ymxw)^2 ;
            reg = nuclearnorm(X_Trn*diag(w)) ;
            obj(iter) = 0.5*err + lambda*reg ;
            disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
                ',rank=' num2str(rank(X_Trn*diag(w))) ',stopALM=' num2str(stopC,'%2.3e') ...
                ',err=' num2str(err) ',norm=' num2str(reg) ',obj=' num2str(obj(iter)) ]);
        end
        
        if stopC<tol
            break;
        else
            Y = Y + mu*leq;
%             if max( max(abs(w-w_old)) , max(max(abs(Z-Z_old))) ) < tol2
%                 rho = rho0 ;
%             else
%                 rho = 1 ;
%             end
            mu = min(max_mu,mu*rho);
%             mu = min(max_mu,mu*rho(iter));
        end 
    end
    
    Coeff(:,i) = w ;    
    
%     figure(2)
%     plot(obj)    
%     figure(1)
%     pause
%     savename = 'Convergence_MNIST.mat' ;
%     save(savename,'obj')    
%     fprintf('the %dth sample is done!\n',i) ;
    
end



