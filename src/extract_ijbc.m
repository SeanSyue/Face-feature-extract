function extract_ijbc(matCaffe, protoTxt, caffeModel, storage_dir, ijbc_data)

    %=============== caffe setttings ================

    addpath(genpath(matCaffe));

    gpu_id = 0;
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    caffe.reset_all();

    net = caffe.Net(protoTxt, caffeModel, 'test');

    %===============extract facescrub_images features and save================
    
    cd(ijbc_data)
    imgNames = dir; % Get names in facescrub dataset
    mkdir(fullfile(storage_dir, 'ijbc_images'));

    % Iterate through one name folder, and extract and save feature
    for j =3:length(imgNames)

        feature = single(zeros(1,512));
        feature(1,:) =  transpose(single(extractDeepFeature(imgNames(j).name, net)));
    
        featName = strrep(imgNames(j).name,'.jpg','.mat')
        featPath = fullfile(storage_dir, 'ijbc_images', featName)
        save(featPath, 'feature');
    end
    cd(ijbc_data) % cd to top level and go to next name folder
end


function feature = extractDeepFeature(file, net)
    img     = single(imread(file));
    img = imresize(img,[112 96]);
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    feature = single(res{1});    
end
