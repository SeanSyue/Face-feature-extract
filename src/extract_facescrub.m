function extract_facescrub(matCaffe, protoTxt, caffeModel, storage_dir, facescrub_data)

    %=============== caffe setttings ================

    addpath(genpath(matCaffe));

    gpu_id = 0;
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    caffe.reset_all();

    net = caffe.Net(protoTxt, caffeModel, 'test');

    %===============extract facescrub_images features and save================
    
    cd(facescrub_data)
    facescurb_names = dir; % Get names in facescrub dataset

    for i = 3:length(facescurb_names) % avoid using the first ones
        imgNameDir = facescurb_names(i).name % Get the current subdirectory name 
        mkdir(fullfile(storage_dir, 'facescrub_images', imgNameDir));

        cd(imgNameDir) % change the directory (then cd('..') to get back)
        imgNames = dir; % Get the file list in the subdirectory

        % Iterate through one name folder, and extract and save feature
        for j =3:length(imgNames)

            feature = single(zeros(1,512));
            feature(1,:) =  transpose(single(extractDeepFeature(imgNames(j).name, net)));
        
            featName = strrep(imgNames(j).name,'.png','.mat')
            featPath = fullfile(storage_dir, 'facescrub_images', imgNameDir, featName)
            save(featPath, 'feature');
        end
        cd(facescrub_data) % cd to top level and go to next name folder
    end
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
