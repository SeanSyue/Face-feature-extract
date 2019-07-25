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

% model   = 'models/sphereface_model/sphereface_deploy.prototxt';
% weights = 'models/sphereface_model/sphereface_model.caffemodel';

model   = 'models/AMSoftmax_model/face_deploy.prototxt';
weights = 'models/AMSoftmax_model/face_train_test_iter_30000.caffemodel';
net     = caffe.Net(model, weights, 'test');

fid  = fopen('IJBC-affine-112X96-lst');
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
%feature = transpose(feature); % seems that you don't need to transpose it again 
fprintf('feature extraction done\n');
fclose(fid);

% end
fprintf('Saving mat ... \n');
% save('sphereface-ijbc-affine-112X96.mat', 'feature')
save('AMSoftmax-ijbc-affine-112X96-h5py.mat', 'feature', '-v7.3');
fprintf('Finished\n');
% end


function feature = extractDeepFeature(file, net)
    img     = single(imread(file));
    % img = imresize(img,[112 96]);
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    feature = single(res{1});    
end

