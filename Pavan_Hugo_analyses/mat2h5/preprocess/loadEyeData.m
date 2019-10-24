function eyeout = loadEyeData( filename )
%function eyeout = loadEyeData( filename )
%        
%Note: filename should have no extension
% 
% eyeout is 2 X samples where the sampling rate is 3 kHz.
%
if filename(end-3) == '.', filename = filename(1:end-4); end

%---------------Load Data Files------

sprintf('%s', 'Loading ns5 eye data, please be patient...')
nsxfile = read_nsx(strcat(filename, '.ns5'), 'chanindx', 1:2);
nsxfile.data = nsxfile.data(:, 1:10:end); %downsample from 30 kHz to 3 kHz
sprintf('%s', 'Loading nev event data...')
nevfile = NEV_reader(strcat(filename, '.nev'));
% 
codes = char(nevfile(find(nevfile(:,1)==0),2)-256)'; %get event codes

%---------------Find calibration parameters---------------

codeIdx = strfind(codes, 'calibPixX');
calibPixX =  regexp(codes(codeIdx:end), '-?\d*', 'match');
calibPixX = str2double(calibPixX(1:9));

codeIdx = strfind(codes, 'calibPixY');
calibPixY =  regexp(codes(codeIdx:end), '-?\d*', 'match');
calibPixY = str2double(calibPixY(1:9));

codeIdx = strfind(codes, 'calibVoltX');
calibVoltX = regexp(codes(codeIdx:end), '-?\d*\.\d*', 'match');
calibVoltX = str2double(calibVoltX(1:9));

codeIdx = strfind(codes, 'calibVoltY');
calibVoltY = regexp(codes(codeIdx:end), '-?\d*\.\d*', 'match');
calibVoltY = str2double(calibVoltY(1:9));

codeIdx = strfind(codes, 'pixPerCM');
pixPerCM = regexp(codes(codeIdx:end), '\d*\.\d*', 'match');
pixPerCM = str2double(pixPerCM(1));

codeIdx = strfind(codes, 'screenDistance');
screenDistance = regexp(codes(codeIdx:end), '\d*', 'match');
screenDistance = str2double(screenDistance(1));


%---------------------------------------------------------

calibration{1}=[calibPixX' calibPixY']; % Pixel space coordinates
calibration{2}=[calibVoltX' calibVoltY']; % Voltage space coordinates
% Volt to pix transform matrices via assumed linear regression
calibration{3}=regress(calibration{1}(:,1),[calibration{2},ones(size(calibration{2},1),1)]); % X
calibration{4}=regress(calibration{1}(:,2),[calibration{2},ones(size(calibration{2},1),1)]); % Y

eyex_row = 1; %index of x position data in file
eyey_row = 2; %index of y position data in file

eyex = nsxfile.data(eyex_row, :);
eyey = nsxfile.data(eyey_row, :);

clearvars nevfile codes 

%--------------set typical Wins Settings--------------
wins.pixelsPerMV = [25, 25];
wins.midV = [250, 250];
wins.voltageDim = [0, 0, 500, 500];
%----------------------------------------------------

eyePt = [eyex' eyey']./1000; %convert millivolts to volts
eyePt = bsxfun(@plus,bsxfun(@times,eyePt,wins.pixelsPerMV),wins.midV);
eyePt(:,2) = bsxfun(@minus,wins.voltageDim(4),eyePt(:,2));
eyePt = cat(2,eyePt,ones(size(eyePt,1),1));
eyex =eyePt*calibration{3};
eyey =eyePt*calibration{4};

%Convert pixels to redefine eyex and eyey in degree space
eyex = pix2deg(eyex, screenDistance, pixPerCM);
eyey = pix2deg(eyey, screenDistance, pixPerCM);

eyeout(eyex_row,:) = eyex;
eyeout(eyey_row,:) = eyey;

end

function dva = pix2deg(pix,scrd,pixpercm)

d = pix./pixpercm;
angle = atan(d./scrd);

dva = (180/pi) * angle;

end


