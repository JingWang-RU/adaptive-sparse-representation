
function split_generation(dataset, nTrain, num_splits)
% function: generate the split
% Input: 
% dataset: e.g. Yale, AR
% nTrain: number of training samples per class
% num_splits: number of reperations of the split

% Output:
% save the index of training and testing samples to savefile

data_fold = ['./Data/' dataset '/'] ;
savefile = ['idxData' num2str(nTrain)];
% load original data
load([data_fold dataset '.mat']);
%¡¡random split

nClass = length( unique( gnd ) ) ;
idxTrain = zeros(num_splits, nTrain*nClass) ;
idxTest = zeros(num_splits, length(gnd) - nTrain*nClass) ;
for s = 1 : num_splits
    idxTrain_s = [] ;
    idxTest_s = [] ;
    flag = 0 ;
    for k = 1 : nClass
        ind = find( gnd == k ) ;
        lenk = length( ind ) ;
        tem = randperm(lenk);
        b = sort(tem(1:nTrain)) + flag ;
        idxTrain_s = [idxTrain_s b] ;
        b = sort(tem(nTrain+1:end)) + flag ;
        idxTest_s = [idxTest_s b] ;        
        flag = flag + lenk ;
    end
    idxTrain(s,:) = idxTrain_s ;
    idxTest(s,:) = idxTest_s ;
end

save( [data_fold savefile], 'idxTrain', 'idxTest');






