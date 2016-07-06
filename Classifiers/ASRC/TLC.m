function [rate predictlabel] = TLC( trnX, trnY, tstX, tstY )

% Reference: Jing Wang, Canyi Lu, et al., "Robust Face Recognition via Adaptive Sparse Representation,"
% IEEE Transactions on System, Man and Cybernetics-Part B, 2014.
% Created by Canyi Lu (canyilu@gmail.com) and Jing Wang (jw998@rutgers.edu).
% Please cite our paper if you use the code, thank you.

% Input:
% trnX [dim * num ] - each column is a training sample
% trnY [ 1  * num ] - training label 
% tstX
% tstY

% Output:
% rate             - Recognition rate of test sample
% predictlabel     - predict label of test sample

ntrn = size( trnX , 2 ) ;
ntst = size( tstX , 2 ) ;

% normalize
for i = 1 : ntrn
    trnX(:,i) = trnX(:,i) / norm( trnX(:,i) ) ;
end
for i = 1 : ntst
    tstX(:,i) = tstX(:,i) / norm( tstX(:,i) ) ;
end


para = 0.001 ; % can be adjusted

A = TLl2_IRLS( tstX , trnX , para ) ;

% residual plan A: similar to SRC
% [rate predictlabel] = Decision_Residual( trnX ,trnY ,tstX , tstY , A ) ;
[rate predictlabel] = Decision_Coeff( trnX ,trnY ,tstX , tstY , A ) ;%
% residual plan B: maximize the sum of the coefficients of inter-class instances
% rate2 = Decision_Coeff( trnX ,trnY ,tstX , tstY , A ) ;

