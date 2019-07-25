% Caffe settings
matCaffe = fullfile(pwd, '../Face-caffe/matlab');
protoTxt = fullfile(pwd, 'models/sphereface_model/sphereface_deploy.prototxt');
caffeModel = fullfile(pwd, 'models/sphereface_model/sphereface_model.caffemodel');

% Configure dataset directories 
facescrub_data = fullfile(pwd, '../insightface-workspace/megaface_testpack_v1.0/data/facescrub_images');
megaface_data = fullfile(pwd, '../insightface-workspace/megaface_testpack_v1.0/data/megaface_images');
ijbc_data = fullfile(pwd, '../insightface-workspace/IJB_release/IJBC/affine-112X96');

% Configure output feature directory
storage_dir = fullfile(pwd, 'features/sphereface_test');

% Run feature extraction
addpath(genpath('src'));
extract_ijbc(matCaffe, protoTxt, caffeModel, storage_dir, ijbc_data)
extract_facescrub(matCaffe, protoTxt, caffeModel, storage_dir, facescrub_data)
extract_megaface(matCaffe, protoTxt, caffeModel, storage_dir, megaface_data)
quit()
