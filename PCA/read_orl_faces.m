function face_vectors = read_orl_faces(dirpath)
dirpath = 'D:\Academic Works\KAU\EE482-Machine Learning with Applications\SourceCodes\Datasets';
P = [dirpath,'\orl_faces\s'];
%112x92 face size

faces = cell(40,10);
for i=1:40  %40 persons
    img_dir = [P, num2str(i)];
    D = dir(fullfile(img_dir,'*.pgm'));
    for k = 1:numel(D)  %10 images per person
        faces{i,k} = imread(fullfile(img_dir,D(k).name));
%         imshow(faces{i,k})
%         pause
    end
end
% pause
size(faces{1,1});
face_vectors = zeros(10304,400);
ctr = 1;
for i=1:40  %40 persons
    for k = 1:numel(D)  %10 images per person
        vec = double(faces{i,k});
        face_vectors(:,ctr) = vec(:);
        ctr = ctr + 1;
    end
end