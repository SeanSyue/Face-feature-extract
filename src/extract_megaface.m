function test_extract_megaface(matCaffe, protoTxt, caffeModel, storage_dir, megaface_data)

    %=============== caffe setttings ================

    addpath(genpath(matCaffe));

    gpu_id = 0;
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    caffe.reset_all();

    net = caffe.Net(protoTxt, caffeModel, 'test');

    %======================extract megaface_images features and save===========

    cd(megaface_data)
    levelOne = dir; % Get first level directory names in megaface dataset

    for i = 3:length(levelOne)-1 % Exclude `lst` file in the megaface folder
        levelOneDirs = levelOne(i).name
        cd(levelOneDirs)
        levelTwo = dir;

        for j = 3:length(levelTwo)
            imgNameDir = levelTwo(j).name % Get second level directory names in megaface dataset
            mkdir(fullfile(storage_dir, 'megaface_images', levelOneDirs, imgNameDir));

            cd(imgNameDir)
            imgNames = dir; % Get the file list in the subdirectory
            
            for k =3:length(imgNames)

                feature = single(zeros(1,512));
                
                feature(1,:) =  transpose(single(extractDeepFeature(imgNames(k).name, net)));
                featName = strrep(imgNames(k).name,'.jpg','.mat')
                featPath = fullfile(storage_dir, 'megaface_images',  levelOneDirs, imgNameDir, featName);
                save(featPath, 'feature');
            end
            cd('..'); % cd to top level and go to next second-level folder
            
        end
        cd(megaface_data); % cd to top level and go to next first-level folder
    end

quit()
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
