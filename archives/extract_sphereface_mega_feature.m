% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to evaluate the performance of the trained model on LFW dataset.
% We perform 10-fold cross validation, using cosine similarity as metric.
% More details about the testing protocol can be found at http://vis-www.cs.umass.edu/lfw/#views.
% 
% Usage:
% cd $SPHEREFACE_ROOT/test
% run code/evaluation.m
% --------------------------------------------------------

% function extract-AMSoftmax()

clear;clc;close all;
% cd('../')

%% caffe setttings
matCaffe = fullfile(pwd, '../Face-caffe/matlab');
addpath(genpath(matCaffe));

gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
caffe.reset_all();


model   = 'models/sphereface_model/sphereface_deploy.prototxt';
weights = 'models/sphereface_model/sphereface_model.caffemodel';
net     = caffe.Net(model, weights, 'test');

storage_dir = '/home/ee303/SEAN/Face-feature-extract/feature_output_mega_mat';
data = '/home/ee303/SEAN/insightface-workspace/megaface_testpack_v1.0/data/facescrub_images';
megaface_img = '/home/ee303/SEAN/insightface-workspace/megaface_testpack_v1.0/data/megaface_images'


%%===================make facescrub_images names dir=======================
%{
cd(data)
D = dir;
cd(storage_dir)
mkdir 'facescrub_images'
cd('facescrub_images')
for i = 3:length(D)
    disp(D(i).name);mkdir(D(i).name);
end
%}
%=========================================================================

%===============extract facescrub_images features and save================
%{
cd(data)
E = dir;
for k = 3:length(E) % avoid using the first ones
    currD = E(k).name % Get the current subdirectory name
      % Run your function. Note, I am not sure on how your function is written,
      % but you may need some of the following
    cd(currD) % change the directory (then cd('..') to get back)
    fList = dir; % Get the file list in the subdirectory
    feature = single(zeros(1,512));
    for j =3:length(fList)
        %fid = fopen(fList(j).name);
        %line = fgetl(fid);
        feature(1,:) =  transpose(single(extractDeepFeature(fList(j).name, net)));
        %transpose(single(extractDeepFeature(fList(j).name, net)))
   
        cd('..')
        cd(storage_dir);
        cd('facescrub_images');
        cd(currD)
    
        filename = fList(j).name;
        filename = strrep(filename,'.png','.mat')
        save(filename, 'feature');
        cd(data);
        cd(currD);
    end
    cd(data)
end
%}
%==========================================================================

%=============make megaface_images dir=====================================
%{
cd(megaface_img)
F = dir;
cd(storage_dir)
mkdir 'megaface_images'
cd('megaface_images')
for i = 3:length(F)-1
    disp(F(i).name);
    cd(megaface_img)
    cd(F(i).name);
    G = dir;
    cd(storage_dir)
    cd('megaface_images')
    mkdir(F(i).name);
    cd(F(i).name);
    for j = 3:length(G)
        mkdir(G(j).name)
    end
%    cd('..');
    %mkdir(F(i).name);
end
%}

%======================extract megaface_images features and save===========

cd(megaface_img)
F = dir;
for i = 3:length(F)-1
    disp(F(i).name);
    cd(megaface_img)
    cd(F(i).name);
    G = dir;
    for j = 3:length(G)
        currD = G(j).name;
        disp(currD);
        
        
        cd(currD)
        fList = dir; % Get the file list in the subdirectory
        feature = single(zeros(1,512));
        
        for k =3:length(fList)
            
            feature(1,:) =  transpose(single(extractDeepFeature(fList(k).name, net)));
 
            cd(storage_dir);
            cd('megaface_images');
            cd(F(i).name);
            cd(currD);
    
            filename = fList(k).name;
            filename = strrep(filename,'.jpg','.mat');
            save(filename, 'feature');
            cd(megaface_img);
            cd(F(i).name);
            cd(currD);
        end
        cd('..');
        
    end
%    cd('..');
    %mkdir(F(i).name);
end







%==========================================================================
cd('/home/ee303/SEAN/Face-feature-extract');
%D = data; % A is a struct ... first elements are '.' and '..' used for navigation.
%for k = 3:length(D)
%    currD = D(k).name;
%    cd(currD);
%fprintf(D.name);
%end    
%{

fid  = fopen('IJBC-affine-112X112-lst');
%lines = fgets(fid);
%line = split(lines,"\n")
i    = 0;
% feature = cell(469375,512); 
%while ischar(line)
%feature= single(zeros(512,469375));
feature= single(zeros(469375,512));
%for a = 1:10
while ~feof(fid)
    line = fgetl(fid);
    i = i + 1;
    fprintf(line);
    fprintf("\n");
    % feature(i).feature = transpose(extractDeepFeature(line, net));
    %feature(i) = mat2cell(transpose(extractDeepFeature(line, net)), 1, 512);
    feature(i,:) = single(transpose(extractDeepFeature(line, net)));
end
feature = transpose(feature);
fprintf('feature extraction done\n');
fclose(fid);

% end
fprintf('Saving mat ... \n');
% save('sphereface-ijbc-affine-112X96.mat', 'feature')
save('sphereface-ijbc-affine-112X112.mat', 'feature');
fprintf('Finished\n');
% end


%}
function feature = extractDeepFeature(file, net)
    img     = single(imread(file));
    img = imresize(img,[112 96]);
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    feature = single(res{1});    
end

