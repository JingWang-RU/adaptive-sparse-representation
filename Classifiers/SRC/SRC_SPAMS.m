function [A] = SRC_SPAMS(  trnX , trnY , tstX, tstY )
% Reference: J. Wright, et al., "Robust Face Recognition via Sparse Representation,"
% IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, pp. 210-227, Feb 2009.
% This code is written based on the Eq. (22) of the reference.

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

% classify
param.lambda = 0.001 ; % not more than 20 non-zeros coefficients
%     param.numThreads=2; % number of processors/cores to use; the default choice is -1
% and uses all the cores of the machine
param.mode = 1 ;       % penalized formulation
param.verbose = false ;       %

A = mexLasso( tstX , trnX , param ) ;


% [rate predictlabel] = Decision_Residual( trnX ,trnY ,tstX , tstY , A ) ;

