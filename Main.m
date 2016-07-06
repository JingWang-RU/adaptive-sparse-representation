
% Reference: Jing Wang, Canyi Lu, Meng Wang, Peipei Li, Shuicheng Yan and Xuegang Hu, "Robust Face Recognition via Adaptive Sparse Representation,"
% IEEE Transactions on System, Man and Cybernetics-Part B, 2014.
% Created by Canyi Lu (canyilu@gmail.com) and Jing Wang (jw998@rutgers.edu)
clear ;
close all;
currentpath = cd ;
AddedPath = genpath( currentpath ) ;
addpath( AddedPath ) ;
fprintf('\n\n**************************************   %s   *************************************\n' , datestr(now) );
fprintf( [ mfilename(currentpath) ' Begins.\n' ] ) ;
fprintf( [ mfilename(currentpath) ' is going, please wait...\n' ] ) ;

%% data set
Data = 'AR' ;                       % AR 100 classes each with about 14 samples
% Data = 'Yale' ;                        % Yale 15 classes each with 11 samples
% Data = 'ORL' ;                    % ORL 40 classes each with 10 sampless
% Data = 'UMIST' ;                 % UMIST 20 classes each with about 29 samples (575 in sum)
% Data = 'YaleB' ;                  % YaleB 38 classes each with about 64 samples

% dimention reduction methods
dr_method = 'PCA' ;
% dr_method = 'LDA' ;
% dr_method = 'Random' ;
% dr_method = 'Identity' ;


%% classifier
Classifier = 'TLC' ;
% Classifier = 'SRC_SPAMS' ;     % SRC (L1)
%  Classifier = 'kNN' ;              % kNN
%  Classifier = 'NFS' ;              % NFS
% Classifier = 'OrthonormalL2' ;    % L2
% Classifier = 'libsvm_test' ;
%  Classifier = 'BSSC';
% Classifier = 'WSRC_SPAMS';

tic;
% KNN NFS LR
%% repeat the experiments setting
splits = 1 ; % can be tuned
%  number of training samples per class
train_num = 4;
switch Data
    case 'Yale'
        Train = train_num : train_num ;
        %   D = [10 30 50 60 70 74 75 78 80 82 84 86 88 89  ] ;
        %   D = [ 5 10 15 20 25 30 40 50 60 70 ] ;  % train_num=5
        %   D = [ 10 30 50 70 90 104];%                % train_num=7
        %   D = [ 10 20 30 40 50 60 70 89 ];           % train_num=6
        D = [ 10 20 30 40 50 59] ;                    % train_num=4
        if strcmp( dr_method , 'LDA' )
            D = [ 3 5 7 9 11 13 14 ] ;                  % maxDim=num class-1
        end
    case 'ORL'
        Train = train_num : train_num ;
        %         D = [20,30,50,60,79];                    % train_num=2
        %         D = [30,50,70,90,119];                  % train_num=3
        %         D = [30,60,90,120,159];                % train_num=4
        D = [ 30 50 80 150 199] ;              % train_num=5
        if strcmp( dr_method , 'LDA' )
            D = [ 5 10 20 25 30 35 39] ;
        end
    case 'UMIST'
        Train = train_num:train_num;
        D = [30 50 70 90 110 ] ;              % train_num=6
        %     D = [ 30 80 130 180 237] ;          % train_num=12
        
        if strcmp( dr_method , 'LDA' )
            D = [ 3:4:19] ;
        end
    case 'YaleB'
        Train = train_num : train_num ;
        D = [ 30 56 120 504] ;% train_num=30/1024
        %    D = [ 30 56 ] ;
        if strcmp( dr_method , 'LDA' )
            D = [5:5:37 ] ;
        end
    case 'AR'
        Train = train_num : train_num ;
        D = [30,50,120,179];               % train_num=2
%         D = [30,54,130,540];                   % train_num=7
        if strcmp( dr_method , 'LDA' )
            Train = train_num ;
            %             D = [ 30 56 99] ;
            %             D = [ 10 20 30 40 50 60 70 80 90] ;
            D = [ 20 40 60 80 99] ;
        end
end
length_D = length(D) ;

%% result file
resultfold = ['.\Results\' Data '\'];
if ~exist(resultfold, 'dir')
    mkdir(resultfold);
end
ResultsTxt = [resultfold  num2str(Train) 'Train_' Classifier '_' dr_method '_D=[' num2str(D) ']_s=' num2str(splits) '.txt' ] ;
fid = fopen( ResultsTxt , 'wt' ) ;
fid = 1 ;
fprintf( fid , '\n\n**************************************   %s   *************************************\n' , datestr(now) );
fprintf( fid , ['Function                   = ' mfilename(currentpath) '.m\n' ] ) ;
fprintf( fid ,  'Data                         = %s\n' , Data ) ;
fprintf( fid ,  'dimension method    = %s\n' , dr_method ) ;
fprintf( fid ,  'Classifier                  = %s\n' , Classifier ) ;
fprintf( fid ,  'splits                        = %d\n\n' , splits ) ;

%% data normalization
path_data = ['.\Data\' Data '\' ] ;
load( [path_data, Data] ) ;
for i = 1 : size(fea,2)
    fea(:,i) = fea(:,i) / norm( fea(:,i) ) ;
end

%% result
for ii = 1: length( Train )
    i = Train( ii ) ;                           % each training sample
    fprintf( fid , 'Train_%d : \n' , i ) ;
    Accuracy = size( splits , length_D ) ;
    load( [path_data 'idxData' num2str(i)] ) ;
    for s = 1 : splits                          % for each data split        
        fea_Train = fea( : , idxTrain(s,:) ) ;
        gnd_Train = gnd( idxTrain(s,:) ) ;
        fea_Test = fea( : , idxTest(s,:) ) ;
        gnd_Test = gnd( idxTest(s,:) ) ;
        [fea_Train,gnd_Train,fea_Test,gnd_Test] = Arrange(fea_Train,gnd_Train,fea_Test,gnd_Test) ;
        
        [ Yfea_Train , Yfea_Test redDim] = DimensionReduction( dr_method , fea_Train , gnd_Train , fea_Test ) ;
        
        for dd = 1 : length_D
            d = D(dd) ;
            tic
                  Accuracy(s,dd) = eval( [ Classifier '( Yfea_Train(1:d,:) , gnd_Train , Yfea_Test(1:d,:) , gnd_Test )' ] ) ;
            toc
            fprintf('********************************************************************\n') ;
            fprintf('dim = %d accu = %6.4f\n', d , Accuracy(s,dd) ) ;
        end
    end
    
    ave_Acc = mean( Accuracy , 1 ) ;
    [max_Acc,dd] = max( ave_Acc ) ;
    max_Dim = D( dd ) ;
    std_Acc = std( Accuracy(:,dd) ) ;
    
    %% save result to txt
    fprintf( fid , 'Dim =\t\t' ) ;
    for dd = 1 : length_D
        fprintf( fid , '\t%5d' , D(dd) ) ;
    end
    fprintf( fid , '\n' ) ;
    for s = 1 : splits
        fprintf( fid, 's = %2d\t%8s ' , s , dr_method ) ;
        for dd = 1 : length_D
            d = D(dd) ;
            fprintf( fid , '\t%.2f ' , Accuracy(s,dd)*100 ) ;
        end
        fprintf( fid , '\n' ) ;
    end
    fprintf( fid , 'ave_Acc %8s ' , dr_method ) ;
    for dd = 1 : length_D
        d = D(dd) ;
        fprintf( fid , '\t%.2f ' , ave_Acc(dd)*100 ) ;
    end
    fprintf( fid , '\n' ) ;
    fprintf( fid , '%dTrain max_Acc¡Àstd_Acc = %.2f¡À%.2f , max_Dim = %d\n' , i , max_Acc*100 , std_Acc*100 , max_Dim  ) ;
    %     fprintf( fid , '%dTrain data is done!\n',i) ;
end

toc

fprintf('\n') ;
if fid ~= 1
    fclose(fid) ;
end
fprintf( [ mfilename(currentpath) ' is done!\n' ] ) ;
% rmpath( AddedPath ) ;
