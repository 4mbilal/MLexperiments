clear all
close all
clc
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
    end
end
% pause
size(faces{1,1})
face_vectors = zeros(10304,400);
ctr = 1;
for i=1:40  %40 persons
    for k = 1:numel(D)  %10 images per person
        vec = double(faces{i,k});
        face_vectors(:,ctr) = vec(:);
        ctr = ctr + 1;
    end
end
face_avg = mean(face_vectors,2);
imshow(uint8(reshape(face_avg,112,92)))

ctr = 1;
for i=1:40  %40 persons
    for k = 1:numel(D)  %10 images per person
        face_vectors(:,ctr) = face_vectors(:,ctr) - face_avg;
        ctr = ctr + 1;
    end
end

S = face_vectors'*face_vectors;
[eigvec,eigval] = eig(S);
eigvec = face_vectors*eigvec;
for k=1:400
    eigvec(:,k) = eigvec(:,k)/sqrt(sum(eigvec(:,k).^2));
end
x=diag(eigval);
[xc,xci]=sort(x,'descend');% largest eigenval
eigvec = eigvec(:,xci);

% [eigvec,mu,eigval] = pca( face_vectors );
figure
plot(xc);

eigenfaces = [];
for k=1:400
    ef  = eigvec(:,k);
    eigenfaces{k} = reshape(ef,112,92);
end

z  = [ eigenfaces{1}  eigenfaces{2}   eigenfaces{3} ; eigenfaces{4}   eigenfaces{5}   eigenfaces{6}];
figure
imshow(z,[],'Initialmagnification','fit');;title('eigenfaces')
% pause

nsel=25;      % Number of eigen faces
ctr = 1;
face_vectors_compressed = zeros(nsel,400);
for i=1:40  %40 persons
    for k = 1:numel(D)  %10 images per person
        for j=1:nsel
            face_vectors_compressed(j,ctr) = sum(eigvec(:,j).*face_vectors(:,ctr));
        end
        ctr = ctr + 1;
    end
end

figure
face_vectors_reconstructed = zeros(10304,400);
for i=1:1  %40 persons
    for k = 9:10  %10 images per person
        ctr = ((i-1)*10)+k;
        for j=1:nsel
            face_vectors_reconstructed(:,ctr) = face_vectors_reconstructed(:,ctr) + face_vectors_compressed(j,ctr)*eigvec(:,j);
        end
        face_vectors_reconstructed(:,ctr) = face_vectors_reconstructed(:,ctr) + face_avg;
        img = reshape(face_vectors_reconstructed(:,ctr),112,92);
        subplot(1,3,1)
        imshow(uint8(img))
        subplot(1,3,2)
        imshow(faces{i,k})
        subplot(1,3,3)
        diff = abs(img-double(faces{i,k}));        
        imshow(uint8(diff))
        title(num2str(sum(sum(diff))/(10304)))
        drawnow
        pause
    end
end
% pause
figure
face_vectors_train = zeros(nsel,40);   %nsec coefficients per person
for i=1:40  %40 persons
    for k = 1:7   %First 7 images per person for training (average)
        ctr = ((i-1)*10)+k;
        for j=1:nsel
            face_vectors_train(j,i) = face_vectors_train(j,i) + (1/7)*face_vectors_compressed(j,ctr);
        end
    end
end

%Testing
False_results = 0;
for i=1:40  %40 persons
    for k = 8:10   %Last 3 images per person for testing
        ctr = ((i-1)*10)+k;
            test = face_vectors_compressed(:,ctr);
            mindist = inf;
            index = 1;
            for m=1:40
               dist = sum((face_vectors_train(:,m)-test).^2);
               if(dist<mindist)
                   mindist = dist;
                   index = m;
               end
            end
             if(index==i)
                 False_results = False_results;
             else
                 False_results = False_results + 1;
             subplot(1,2,1)
             imshow(faces{i,k})
             title('Test Face')
             subplot(1,2,2)
             imshow(faces{index,1})
             title('Confused with..')
             drawnow
             pause
             end
    end
end

False_results*100/120
