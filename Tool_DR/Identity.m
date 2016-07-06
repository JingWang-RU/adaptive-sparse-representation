function [ eigvector , eigvalue ] = Identity( fea_Train )

% 

dim = size( fea_Train , 1 ) ;
eigvector = eye( dim , dim ) ;
eigvalue = zeros( dim , 1 ) ;
