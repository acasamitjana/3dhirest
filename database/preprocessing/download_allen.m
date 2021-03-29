clear all

ATLAS_ID='265297126';
GROUPS='31,113753816,141667008,265297118'; % Brodmann
DS_FACTOR=5; % in -log2! 0 is 1micron, 9 is 512 micron
datadir='/home/acasamitjana/Data/Allen/downloads/';
secInfoMatFile=[datadir '/secInfo.mat'];

if exist(datadir,'dir')==0
    mkdir(datadir);
end

% if exist(secInfoMatFile,'file')
%
%     load(secInfoMatFile,'annotatedNissl','secIDNissl','secNumberNissl', 'secIDNissl','secNumberIHC', 'DS_FACTOR');
%
% else

disp('Downloading info for all sections');

url = 'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::SectionDataSet,rma::criteria,specimen[id$eq708424],rma::include,section_images(associates,alternate_images,treatments)';
data=urlread(url);
disp('Building list of sections');

secNumberNissl=zeros(1,10000);
secIDNissl=zeros(1,10000);
annotatedNissl=zeros(1,10000);
secNumberIHC=zeros(1,10000);
secIDIHC=zeros(1,10000);

indNissl=1;
indIHC=1;
ready=0;
[~,rem]=strtok(data,char(10));
datasetID={};
datasetImages=0;
insideAtlasImage=0;
insideTreatment = 0;
secID = 0;
secNumber = 0;
it_tok=0;
while ready==0
    [tok,rem]=strtok(rem,char(10));
    it_tok = it_tok+length(tok);
    if isempty(tok)
        ready=1;

    elseif contains(tok,'<section-image>')% ~isempty(strfind(tok,'<section-image>'))
        insideAtlasImage=1;


    elseif contains(tok,'</section-image>')%~isempty(strfind(tok,'</section-image>'))
        insideAtlasImage=0;
        secID = 0;
        secNumber = 0;

    elseif contains(tok,'<treatment>')%~isempty(strfind(tok,'<treatment>'))
        insideTreatment=1;

    elseif contains(tok,'</treatment>')%~isempty(strfind(tok,'</treatment>'))
        insideTreatment=0;

    elseif insideAtlasImage && contains(tok,'<id type="integer">') && secID==0%~isempty(strfind(tok,'<id type="integer">'))
        f1=find(tok=='<');
        f2=find(tok=='>');
        secID=str2double(tok(f2(1)+1:f1(2)-1));

    elseif insideAtlasImage && contains(tok,'<section-number type="integer">') && secNumber==0%~isempty(strfind(tok,'<section-number type="integer">'))
        f1=find(tok=='<');
        f2=find(tok=='>');
        secNumber=str2double(tok(f2(1)+1:f1(2)-1));

    elseif insideAtlasImage &&  contains(tok,'<annotated type="boolean">')%~isempty(strfind(tok,'<annotated type="boolean">'))
        f1=find(tok=='<');
        f2=find(tok=='>');
        if strcmp(strtrim(lower(tok(f2(1)+1:f1(2)-1))),'true')
            annotated=1;
        else
            annotated=0;
        end

    elseif insideAtlasImage && insideTreatment && contains(tok,'<id type="integer">')%~isempty(strfind(tok,'<id type="integer">'))
        f1=find(tok=='<');
        f2=find(tok=='>');
        id_treatment = str2double(tok(f2(1)+1:f1(2)-1));

        if id_treatment == 3 && secIDNissl(indNissl)==0
            secIDNissl(indNissl) = secID;
            secNumberNissl(indNissl) = secNumber;
            annotatedNissl(indNissl) = annotated;
            indNissl=indNissl+1;

        elseif id_treatment == 16 && secIDIHC(indIHC)==0
            secIDIHC(indIHC) = secID;
            secNumberIHC(indIHC) = secNumber;
            indIHC=indIHC+1;
        end

    end
end


secIDIHC=secIDIHC(1:indIHC-1);
secNumberIHC=secNumberIHC(1:indIHC-1);

secIDNissl=secIDNissl(1:indNissl-1);
secNumberNissl=secNumberNissl(1:indNissl-1);
annotatedNissl=annotatedNissl(1:indNissl-1);

% nicer if they're sorted
[secNumberNissl,idx]=sort(secNumberNissl);
annotatedNissl=annotatedNissl(idx);
secIDNissl=secIDNissl(idx);

[secNumberIHC,idx]=sort(secNumberIHC);
secIDIHC=secIDIHC(idx);

save(secInfoMatFile, 'annotatedNissl','secIDNissl','secNumberNissl', 'secIDIHC','secNumberIHC', 'DS_FACTOR');


% end
disp('Downloading images and segmentations')
disp('NISSL')
for i=1:length(secNumberNissl)
    imfile=[datadir filesep 'nissl' filesep 'images_orig' filesep 'image_' num2str(secNumberNissl(i),'%.4d') '.jpg'];
    segfile=[datadir filesep 'nissl' filesep 'labels_orig' filesep 'seg_' num2str(secNumberNissl(i),'%.4d') '.svg'];
    disp(['Section ' num2str(i) ' of ' num2str(length(secNumberNissl))]);

    if exist(imfile,'file')
        disp('  Image already downloaded');
    else

        disp('  Downloading image');
        % Image: it's better to download a bit higher res and downsample
        url=['http://api.brain-map.org/api/v2/image_download/' num2str(secIDNissl(i)) '?downsample=' num2str(DS_FACTOR-1) '&quality=100'];
        urlwrite(url,'/tmp/kk.jpg');
        aux=imread('/tmp/kk.jpg');
        aux=imresize(aux,floor([size(aux,1)/2 size(aux,2)/2]));
        imwrite(aux,imfile,'Quality',100);
    end

    if annotatedNissl(i)==0
        disp('  Annotations not available for this section');
    elseif exist(segfile,'file')
        disp('  Annotations already downloaded');
    else
        disp('  Downloading annotations');
        url=['http://api.brain-map.org/api/v2/svg_download/' num2str(secIDNissl(i)) '?groups=' GROUPS '?downsample=' num2str(DS_FACTOR)];
        urlwrite(url,segfile);
    end

end

%Generate masks
for i=1:length(secNumberNissl)
    disp(['NISSL: ' num2str(i) '/' num2str(length(secNumberNissl))])
    I = imread([NISSL_DIR filesep 'images_orig' filesep 'image_' num2str(secNumberNissl(i),'%04d' ) '.jpg']);

    I = rgb2gray(I);

    Imedian = medfilt2(I);
    Iaverage = filter2(fspecial('average',3),Imedian);
    Maverage = double(Iaverage < 248);
    if it_slice == 166 || it_slice == 222
        Maverage = double(Iaverage < 240);
    end
    Maverage(1:10,:) = 0;
    Maverage(end-10:end,:) = 0;
    Maverage(:,1:10) = 0;
    Maverage(:,end-10:end) = 0;

    CC = bwconncomp(Maverage);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [pixels_sorted, idx_sorted] = sort(numPixels, 'descend');
    MCC = zeros(size(Maverage));
    for it_s = 1: sum(pixels_sorted > 30000)
        MCC(CC.PixelIdxList{idx_sorted(it_s)}) = 1;
    end

    v=[-1 0 1];
    Gx = imfilter(Iaverage,reshape(v,[3 1]));
    Gy = imfilter(Iaverage,reshape(v,[1 3]));
    Gm = sqrt(Gx.*Gx+Gy.*Gy);
    M = Gm > 1;

    Mnew = MCC.*M;
    se = strel('square', 10);
    Mfill = imfill(Mnew, 'holes');
    Mop = imopen(Mfill,se);


    CC = bwconncomp(Mop);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [pixels_sorted, idx_sorted] = sort(numPixels, 'descend');
    MCCop = zeros(size(M));
    for it_s = 1: sum(pixels_sorted > 30000)
        MCCop(CC.PixelIdxList{idx_sorted(it_s)}) = 1;
    end

    imwrite(MCCop, [datadir filesep 'nissl' filesep 'masks_orig' filesep 'image_' num2str(secNumberNissl(i),'%.4d' ) '.png' ])


end


disp('IHC')
for i=1:length(secNumberIHC)
    imfile=[datadir filesep 'ihc' filesep 'images_orig' filesep 'image_' num2str(secNumberIHC(i),'%.4d') '.jpg'];
    disp(['Section ' num2str(i) ' of ' num2str(length(secNumberIHC))]);

    if exist(imfile,'file')
        disp('  Image already downloaded');
    else

        disp('  Downloading image');
        % Image: it's better to download a bit higher res and downsample
        url=['http://api.brain-map.org/api/v2/image_download/' num2str(secIDIHC(i)) '?downsample=' num2str(DS_FACTOR-1) '&quality=100'];
        urlwrite(url,'/tmp/kk.jpg');
        aux=imread('/tmp/kk.jpg');
        aux=imresize(aux,floor([size(aux,1)/2 size(aux,2)/2]));
        imwrite(aux,imfile,'Quality',100);
    end

end


%Generate masks
for i=1:length(secNumberIHC)
    disp(['IHC: ' num2str(it_slice) '/' num2str(length(secNumberIHC))])
    I = imread([datadir filesep 'ihc' filesep 'images_orig' filesep 'image_' num2str(secNumberIHC(it_slice),'%04d' ) '.jpg']);

    if it_slice < 10
        min_size = 10000;
    elseif it_slice < 150
        min_size = 60000;
    else
        min_size = 30000;
    end
    I = rgb2gray(I);

    Imedian = medfilt2(I);
    if it_slice < 148
        Iaverage = filter2(fspecial('average',3),Imedian);
        Maverage = double(Iaverage < 250);
    else
        Iaverage = filter2(fspecial('average',10),Imedian);
        Maverage = double(Iaverage < 245);

    end


    Maverage(1:10,:) = 0;
    Maverage(end-10:end,:) = 0;
    Maverage(:,1:10) = 0;
    Maverage(:,end-10:end) = 0;


    se = strel('square', 5);
    MCCfill = imfill(Maverage, 'holes');
    MCCfillop = imopen(MCCfill,se);

    CC = bwconncomp(MCCfillop);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [pixels_sorted, idx_sorted] = sort(numPixels, 'descend');
    MCC = zeros(size(Maverage));
    for it_s = 1: sum(pixels_sorted > min_size)
        MCC(CC.PixelIdxList{idx_sorted(it_s)}) = 1;
    end
    MCCop = imopen(MCC,se);

    imwrite(MCCop, [datadir filesep 'ihc' filesep 'masks_orig' filesep 'slice_' num2str(secNumberIHC(it_slice),'%03d' ) '.png' ])

end