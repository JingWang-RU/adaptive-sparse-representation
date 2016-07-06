function [ fea_Train, fea_Test , redDim ] = DimensionReduction( method , fea_Train , gnd_Train , fea_Test )

% method        'PCA','LDA,'Random'
% fea_Train     dim*num_Train 
% gnd_Train     1*num_Train(or num_Train*1)
% parameter    

% Dimension reduction
switch method
    case 'PCA'
        [ eigvector , eigvalue ] = PCA( fea_Train ) ;
    case 'LDA'
        options.PCARatio = 1 ;
        [ eigvector , eigvalue ] = LDA( gnd_Train , options , fea_Train ) ;       
    case 'Random'
        dim = size( fea_Train , 1 ) ; 
        eigvector = randn( dim , dim ) ;
        eigvalue = zeros(dim,1) ;    
    case 'Identity'
        [ eigvector , eigvalue ] = Identity( fea_Train ) ;  
   
end

fea_Train = eigvector' * fea_Train ;
fea_Test = eigvector' * fea_Test ;
redDim = length(eigvalue) ;
