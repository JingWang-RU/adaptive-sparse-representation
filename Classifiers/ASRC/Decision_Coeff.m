function [accuracy ind] = Decision_Coeff( fea_Train, gnd_Train ,fea_Test , gnd_Test , Coeff_Test )

num_Test = size(fea_Test,2) ;
nClass = length( unique(gnd_Train) ) ;
SA = zeros(nClass,num_Test) ;
for i = 1 : num_Test
   for k = 1 : nClass       
       index = find( gnd_Train == k ) ;
       SA(k,i) = sum( Coeff_Test( index ,i ) ) ;
%    SA(k,i) = norm( Coeff_Test( index ,i ) , 2 ) ;
   end
end
% for i = 1 : num_Test
%     SA(:,i) = SA(:,i)/norm(SA(:,i)) ;
% end
[val ind] = max( SA ) ;
accuracy = sum( ind == gnd_Test ) / num_Test ;
