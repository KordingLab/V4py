function out = read_nsx(filename,varargin)
%
% Based on Ripple's NSX2MAT function - should read any NSX file
%
% Reformatted to be compatible with FieldTrip -ACS (adam@adamcsnyder.com)

%% Ignore MATLAB's complaints about code optimization:
%#ok<*NASGU>
%#ok<*AGROW>
%% optional input arguments

p = inputParser;
p.addRequired('filename',@ischar);
p.addOptional('begsample',1,@isscalar);
p.addOptional('endsample',-1,@isscalar);
p.addOptional('chanindx',-1,@isnumeric);
p.addOptional('readdata',true,@islogical);

p.parse(filename,varargin{:});

begsample = p.Results.begsample;
endsample = p.Results.endsample;
chanindx = p.Results.chanindx;
readdata = p.Results.readdata;

%% open file
fh   = fopen(filename, 'rb');
%% get file size
fseek(fh, 0, 1);
filesize = ftell(fh);
fseek(fh, 0, -1);
%% check if valid nsx 2.2 file
fid  = fread(fh, 8, '*char')';
if ~strcmp(fid, 'NEURALCD') %note: <-might use this for ft_filetype
    error('Not Valid NSx 2.2 file');
end
fseek(fh, 2, 0); %skip file spec
%% get bytes in headers (used to jump to begining of data)
bytesInHeaders = fread(fh, 1, '*uint32');
%% get label
label          = fread(fh, 16, '*char')'; %#ok<NASGU>
fseek(fh, 256, 0);
%% get sampling frequency
period         = fread(fh, 1, '*uint32');
fs             = 30000/period; % samples per second
clockFs        = fread(fh, 1, '*uint32');
fseek(fh, 16, 0); %<-This skips the "resolution of time stamps" and "time origin" parameters of the header.
%% get channel list, unit, and scale
chanCount      = fread(fh, 1, '*uint32');
scale          = zeros(chanCount,1);
channelID      = int16(scale);
fseek(fh, 2, 0);
for i = 1:chanCount
    channelID(i) = fread(fh, 1, '*uint16');
    fseek(fh, 18, 0);
    minD         = fread(fh, 1, 'int16');
    maxD         = fread(fh, 1, 'int16');
    minA         = fread(fh, 1, 'int16');
    maxA         = fread(fh, 1, 'int16');
    unit(i)      = {deblank(fread(fh, 16, '*char')')}; 
    scale(i)        = (maxA - minA)/(maxD - minD);  
    fseek(fh, 22, 0);
end
chanLabels = cellfun(@num2str,num2cell(double(channelID)),'uniformoutput',0);
fseek(fh, bytesInHeaders,-1);
%% get time vector
k              = 1;
while (ftell(fh) < filesize)
    header         = fread(fh, 1); 
    timeStamp(k)   = fread(fh, 1, '*uint32');
    ndataPoints(k) = fread(fh, 1, '*uint32'); 
    fseek(fh,(2*double(ndataPoints(k))*double(chanCount)),0);
    k = k + 1;
end
time               = [timeStamp; timeStamp + ndataPoints.*period]; %changed to ndataPoints.*period. 'timeStamp' was in units of the clock frequency, but ndataPoints was in the data frequency. -ACS 08Nov2012
%% get data vectors
nvec           = double(cumsum(double(ndataPoints)));
if readdata
    nvec           = [0, nvec];
    if endsample<0, endsample=nvec(end);end;
    data           = zeros(chanCount, endsample-begsample+1, 'int16');
    if any(chanindx<0),chanindx=1:chanCount;end;
    fseek(fh, bytesInHeaders, -1);
    bytes2skip = (begsample-1)*2*chanCount; %this should be the number of data samples to skip... -ACS 09May2012
    bytes2skip = bytes2skip+9*sum(begsample>nvec); %this should add on the little headers before each data block... -ACS
    fseek(fh, bytes2skip, 0); %The initial bytes to skip
    dataBlockBounds = nvec>=begsample&nvec<endsample; %pretty sure these are the right booleans here... -ACS
    if ~any(dataBlockBounds) %if there are no block boundaries in the requested data segment
        %just pull out the block of data:
        data = fread(fh,[chanCount,endsample-begsample+1],'*int16');
    else %if there are block boundaries in the requested data segment
        endByte = ftell(fh)+9*sum(dataBlockBounds)+(endsample*2*chanCount); %this should be the position of the last requested sample in the file... -ACS
        currentSample = begsample;
        dataInd = 1;
        while ftell(fh)<endByte
            nextBound = nvec(find(nvec>currentSample,1,'first'));
            data(:,dataInd:nextBound-currentSample+dataInd) = fread(fh,[chanCount,nextBound-currentSample+1],'*int16');
            currentSample = nextBound+1;
            dataInd = size(data,2)+1;
            if ftell(fh)<(filesize-9)
                fseek(fh,9,0); %skip the little header for the next data block
            end;
        end;
    end;
    out.data = bsxfun(@times,double(data(chanindx,:)),double(scale(chanindx))); %this is my way of trimming it for now - will eventually take care of this during reading in step. -ACS 27Apr2012
end;
%% package output
hdr.Fs = fs;
hdr.nChans = chanCount;
hdr.nSamples = max(nvec); %changed to nvec, which is the cumulative sum of datapoints in each block (had been max(ndataPoints), which caused an error for files with more than one block). -ACS 08Nov2012
hdr.label = chanLabels;
hdr.chanunit = unit(:);
hdr.scale = scale;
hdr.timeStamps = double(time);
hdr.clockFs = double(clockFs);

out.hdr = hdr;

fclose(fh);
