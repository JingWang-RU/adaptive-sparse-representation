function [accuracy predictlabel] = NFS( fea_Train ,gnd_Train ,fea_Test , gnd_Test )

% Input:
% fea_Train [dim * num ] - each column is a training sample
% gnd_Train [ 1  * num ] - training label 
% fea_Test
% gnd_Test

% Output:
% rate             - Recognition rate of test sample
% predictlabel     - predict label of test sample

[dim,num_Train] = size( fea_Train ) ;
num_Test = size( fea_Test , 2 ) ;
nClass = length( unique(gnd_Train) ) ;

% normalize
for i = 1 : num_Train
    fea_Train(:,i) = fea_Train(:,i) / norm( fea_Train(:,i) ) ;
end
for i = 1 : num_Test
    fea_Test(:,i) = fea_Test(:,i) / norm( fea_Test(:,i) ) ;
end

Coeff_Test = zeros(num_Train,num_Test) ;

lambda = 0.01 ;

for k = 1 : nClass
    ind = find( gnd_Train == k ) ;
    X = fea_Train(:,ind) ;
    I = lambda * eye(length(ind),length(ind)) ;
    H = ( X'*X + I ) \ X' ; 
%     H = ( X'*X ) \ X' ;
%     [Q,R]=qr(X,0) ;
%     B = R\Q' ;
%     R = pinv(R) ;
%     H = R*Q' ;

    Coeff_Test(ind,:) = H * fea_Test ;
end

% % provided by matlab lsqlin; the results is same as above code
% X = cell(nClass,1) ;
% index = cell(nClass,1) ;
% one = cell(nClass,1) ;
% for k = 1 : nClass
%     index{k} = find( gnd_Train == k ) ;
%     X{k} = fea_Train(:,index{k}) ;
%     one{k} = ones( 1 , length(index{k}) ) ;
% end
% 
% options.LargeScale = 'off' ;
% for i = 1 : num_Test
%     for k = 1 : nClass
%        Coeff_Test(index{k},i) = lsqlin( X{k} , fea_Test(:,i) , [] , [] , one{k}  , 1 , [] , [] , [] , options ) ;
%     end    
% end

[accuracy predictlabel] = Decision_Residual( fea_Train ,gnd_Train , fea_Test , gnd_Test , Coeff_Test ) ;

% accuracy2 = Decision_Coeff( fea_Train ,gnd_Train , fea_Test , gnd_Test , Coeff_Test ) ;
% 
% Coeff_Test = abs(Coeff_Test) ;
% accuracy3 = Decision_Residual( fea_Train ,gnd_Train , fea_Test , gnd_Test , Coeff_Test ) 
% accuracy4 = Decision_Coeff( fea_Train ,gnd_Train , fea_Test , gnd_Test , Coeff_Test ) 


