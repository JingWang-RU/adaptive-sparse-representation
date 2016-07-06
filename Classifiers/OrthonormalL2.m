%  Qinfeng Shi, Anders Eriksson, Anton van den Hengel, Chunhua Shen, Is face 
% recognition really a Compressive Sensing problem? In IEEE Computer Society 
% Conference on Computer Vision and Pattern Recognition (CVPR 11), Colorado 
% Springs, USA, June 21-23, 2011.
 
function Accuracy = OrthonormalL2(  fea_Train , gnd_Train , fea_Test , gnd_Test )
% function [Coeff_Test] = OrthonormalL2(  fea_Train , gnd_Train , fea_Test , gnd_Test )

% Input:
% trnX [dim * num ] - each column is a training sample
% trnY [ 1  * num ] - training label 
% tstX
% tstY

[dim mun_Train] = size( fea_Train ) ;
num_Test = size( fea_Test , 2) ;

% normalize
for i = 1 : mun_Train
    fea_Train(:,i) = fea_Train(:,i) / norm( fea_Train(:,i) ) ;
end
for i = 1 : num_Test
    fea_Test(:,i) = fea_Test(:,i) / norm( fea_Test(:,i) ) ;
end

% [Q,R]=qr(fea_Train,0) ;
% B = R\Q' ;
% R = pinv(R) ;
% B = R*Q' ;
% Coeff_Test = B * fea_Test ;

lambda = 0.001 ;
I = lambda * eye(mun_Train,mun_Train) ;
B = (fea_Train'*fea_Train+I) \ fea_Train' ;  
Coeff_Test = B * fea_Test ;
% k = 0 ;
% Coeff_Test = Selection( Coeff_Test , k ) ;

Accuracy = Decision_Residual( fea_Train , gnd_Train , fea_Test , gnd_Test , Coeff_Test ) ;


