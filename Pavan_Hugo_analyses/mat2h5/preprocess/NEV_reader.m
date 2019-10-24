function [spike, waves] = NEV_reader(nevfile, waveson)
%function [spike, waves] = NEV_reader(nevfile, waveson)
%
% NEV_reader takes an NEV file as input and returns the event codes
% and times. If the flag "waveson" is set to 1, it also returns the
% waveform snippets associated with each timestamp. If waveson is
% not passed in, the default is 0.
%
% The columns of "spike" are channel, spike class (or digital port
% value for digital events), and time (seconds). The channel is 0
% for a digital event, and 1:255 for spike channels. 1:128 are the
% array channels, and 129 is the first analog input.
%
%

if (nargin < 2)
    waveson = 0;
end

%Header Basic Information
fid = fopen(nevfile,'r','l');
identifier = fscanf(fid,'%8s',1); %File Type Indentifier = 'NEURALEV'
filespec = fread(fid,2,'uchar'); %File specification major and minor 
version = sprintf('%d.%d',filespec(1),filespec(2)); %revision number
fileformat = fread(fid,2,'uchar'); %File format additional flags
headersize = fread(fid,1,'ulong'); 
%Number of bytes in header (standard  and extended)--index for data
datapacketsize = fread(fid,1,'ulong'); 
%Number of bytes per data packet (-8b for samples per waveform)
stampfreq = fread(fid,1,'ulong'); %Frequency of the global clock
samplefreq = fread(fid,1,'ulong'); %Sampling Frequency

%Windows SYSTEMTIME
time = fread(fid,8,'uint16');
year = time(1);
month = time(2);
dayweek = time(3);
if dayweek == 0 
    dw = 'Sunday';
elseif dayweek == 1 
    dw = 'Monday';
elseif dayweek == 2 
    dw = 'Tuesday';
elseif dayweek == 3 
    dw = 'Wednesday';
elseif dayweek == 4 
    dw = 'Thursday';
elseif dayweek == 5 
    dw = 'Friday';
elseif dayweek == 6 
    dw = 'Saturday';
end
day = time(4);
date = sprintf('%s, %d/%d/%d',dw,month,day,year);
disp(date);
hour = time(5);
minute = time(6);
second = time(7);
millisec = time(8);
time2 = sprintf('%d:%d:%d.%d',hour,minute,second,millisec);
disp(time2);

%Data Acquisition System and Version
application = fread(fid,32,'uchar')';

%Additional Information (and Extended Header Information)
comments = fread(fid,256,'uchar')';
extheadersize = fread(fid,1,'ulong');

fclose(fid);

%-------------------------------------------------------------------------------
%Read DATA
%Header Basic Information (skip over)
fid = fopen(nevfile,'r','l');
header = fread(fid,headersize,'uchar');

%Data Packets
%---------------------
%indexing
x = 0;
n = 1;
m = 1;
k = 1;
sweep(1:32) = 0;

increment = 1024*1024; % one megabyte

nextthresh = increment;
spike = zeros(0,3);

waves = cell(0);

while x == 0
   [timestamp,c] = fread(fid,1,'ulong');
   if c == 0 x = 1; disp('Finished reading file'); break; end
   electrode = fread(fid,1,'int16');
   class = fread(fid,1,'uchar');
   future = fread(fid,1,'uchar');
   
   %Stimulus Number and start time
    if electrode == 0             
        %signals experimental information
        dig = fread(fid,1,'uint16');
        analog1 = fread(fid,1,'short');
        analog2 = fread(fid,1,'short');
        analog3 = fread(fid,1,'short');
        analog4 = fread(fid,1,'short');
        analog5 = fread(fid,1,'short');
        fread(fid,(datapacketsize-20)/2,'int16');
        
        spike(m,3) = (timestamp/samplefreq);
        spike(m,2) = dig; % value on the digital port
        spike(m,1) = 0; % zero indicates digital event
        m = m + 1;
    else
        % if the spike array is filled up, double the size
        if (size(spike,1) < m)
            spike(2*m,:) = [0 0 0];
            if (waveson)
                waves{2*m} = [];
            end
        end
        
        % only store the wave info if we are asked to - it's too large.
        if (waveson)
            waves{m} = int16(fread(fid,(datapacketsize-8)/2,'int16'));
        else
            fread(fid,(datapacketsize-8)/2,'int16');
        end

        % store the spike times and channels in spike array
        spike(m,3) = (timestamp/samplefreq); %global time (msec)
        spike(m,2) = class; %spike classification
        spike(m,1) = electrode;	%electrode number
        m = m+1;
    end
    start_exp = 1;

    if ftell(fid) > nextthresh
       display(sprintf('File Position: %i MB',nextthresh/increment));
       nextthresh = nextthresh + increment;
    end
end %while loop

% truncate the arrays where they should end - they may have been made large
spike = spike(1:m-1,:);
if (waveson)
  waves = waves(1:m-1);
end
