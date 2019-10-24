function ex = nev2ex(nev, alignCode, diode, thresh)
%function ex = nev2ex(nev)
%function ex = nev2ex(nev, alignCode)
%function ex = nev2ex(nev, alignCode, diode)
%function ex = nev2ex(nev, alignCode, diode, thresh)
%
% Function to read nev files created with Ex and convert to a
% structure with trial information
%
% nev: a Nx3 matrix where the columns are elec num, sort code, time(s)
%
% alignCode: if present, adjusts the times so that a time of 0
%  corresponds to the presence of the sort code given.  Give -1 for
%  aligning to the onset of the diode. If more than one instance of
%  alignCode is found in the trial, the first is used.
%
% WARNING: DIODE FEATURE NOT WELL TESTED!!!!!!!
% diode: the 30kHz diode signal (optional). should call read_nsx and then
% send in the diode signal (usually channel 3) to this function
%
% thresh: the value for the diode threshold.  aligning happens the
%  first time the diode is greater than this value. If thresh is not
%  specified, it uses 75% of the max value in the diode signal
%
% output is a struct with the fields:
%   EVENTS: spike times, in seconds, unit x condition x sort code
%   CODES: digital codes and timestamps (s), unit x condition x
%   sort code
%   CHANNELS: List of units, elec num x sort code
%   AND MORE ...
%
% NOTE: This code currently uses the reward code (5) to determine
% whether to keep a trial. It would need to be changed if you
% wanted to include unrewarded trials. If no reward codes are
% found, it uses the correct code (150).
%

diodeSamp = 30000;
diodeTimeStep = 1/diodeSamp;

START_TRIAL = 1;
END_TRIAL = 255;
REWARD = 5;
CORRECT = 150;

if nargin > 1
    alignFlag = 1;
else
    alignFlag = 0;
end

if nargin < 3
    diode = [];
end

if nargin < 4
    thresh = max(diode) * 0.75;
    %thresh = 300;
end

warnVarFlag = 0;
warnCodeFlag = 0;

% flag to include 0 and 255 codes in output. By default, remove these
include_0_255 = 0;

codes = nev(nev(:,1)==0,2:3);
nev = nev(nev(:,1) ~= 0,:);

channels = unique(nev(:,1:2),'rows');
if ~include_0_255
  channels = channels(channels(:,2) ~= 0 & channels(:,2) ~= 255 & channels(:,1) ~= 0,:);
else
  channels = channels(channels(:,1) ~= 0,:);
end

starts = find(codes(:,1) == START_TRIAL);
ends = find(codes(:,1) == END_TRIAL);

if length(starts) ~= length(ends)
    warning([num2str(length(starts)),' Trial starts and ', ...
             num2str(length(ends)),' Trial ends. Continuing.']);
    %if length(starts) - 1 == length(ends)
    %    disp('Warning: One extra start code');
    %    starts = starts(1:end-1);
    %else
    %    error('Something is wrong: Different trial start and end counts');
    %end
end

%if sum((ends-starts)<0) > 0
%    error(['Something is wrong: some trial starts are after the ' ...
%           'corresponding ends']);
%end

trials = cell(length(starts),1);
env = struct;

tstarts = starts;
tcount = 1;
while ~isempty(tstarts) % look through all the start codes

    % find the first end code after the start you're looking at
    nextend = ends(min(find([ends - tstarts(1)] > 0)));
    tcodes = codes(tstarts(1):nextend,:);

    % Double-check there's no more than one end code
    if length(find(tcodes(:,1) == 255)) > 1
        error(['Trial ',num2str(tcount),' had more than 1 end code. Serious error - exiting']);
    end
    
    % check to see if there is an interrupted trial buried here -
    % if so, cut off the codes after the second '1' in the trial so
    % that you can parse just this trial
    if length(find(tcodes(:,1) == 1)) ~= 1
        warning(['Trial ',num2str(tcount),' was apparently interrupted.']);
        % if this is the last trial, exit the loop
        if (length(tstarts)==1)
            trials(end)=[];
            tstarts(1) = []; % delete this trial from the list
            continue
        else
            %tstarts(1) = []; % kill this trial start and go to the next
            tcodes = tcodes(1:find(tcodes(:,1)==1,1,'last')-1,:);
            nextend = tstarts(1)+size(tcodes,1);
        end
    end
    
    trial = struct();
    trial.codes = tcodes;
    %disp(['Codes on trial ',num2str(tcount)]);
    %disp(trial.codes(:,1)');pause;
    cndIndex = find(trial.codes(:,1)>32768,1);
    trial.cnd = trial.codes(cndIndex,1)-32768;
    trial.codes(cndIndex,:) = [];
    
    %startTime = codes(tstarts(1),2);
    %endTime = codes(nextend,2);
    startTime = trial.codes(1,2);
    endTime = trial.codes(end,2);    
    trial.startTime = startTime;
    trial.endTime = endTime;

    % Check codes by start/endtime and compare to indexing method
    % MATT MATT why does startTime/EndTime seem to be off on 1574
    tc=codes(codes(:,2) >= startTime & codes(:,2) <= endTime,:);
    if (size(tc,1) ~= size(tcodes,1))
        error(['Code timing vs. indexing mismatch on trial ',num2str(tcount)]);
    end
    
    trial.spikes = nev(nev(:,3) > startTime & nev(:,3) < endTime,:);

    % adjust times to start of trial
    trial.spikes(:,3) = trial.spikes(:,3) - startTime;
    trial.codes(:,2) = trial.codes(:,2) - startTime;

    if ~isempty(diode)
        % old 1kHz code
        %trial.diode = diode(round(startTime*1000):round(endTime*1000));
        trial.diode = diode(round(startTime*diodeSamp):round(endTime*diodeSamp));
        trial.diodeTime = diodeTimeStep:diodeTimeStep:(length(trial.diode)/diodeSamp);
    end

    trial.channels = channels;
    msgInd = trial.codes(:,1) >= 256 & trial.codes(:,1) < 512;
    trial.msgs = char(trial.codes(msgInd,1)-256)';
    variables = regexp(trial.msgs,';','split');
    for j = 1:length(variables)
        if variables{j}
            try
                % this option is for plain numeric values
                eval(['trial.env.' variables{j} ';']);
            catch
                % if it's not simple numeric, save it as a string
                vt=regexp(variables{j},'=','split');
                try
                    %MATT MATT still not right
                    eval(['trial.env.' char(vt{1}(:))' '=' '''' char(vt{2}(:))' '''' ';']);
                    %eval(['trial.env.' char(vt{1}(1)) '=' '''' char(vt{1}(2)) '''' ';']);
                catch
                    % if that fails, send a warning 
                    %if ~warnVarFlag
                    disp(['Could not parse variable name: ',variables{j}]);
                    %disp(['No more warnings of this type will be displayed']);
                    %   warnVarFlag = 1;
                    %end
                end
            end
        end
    end
    
    trials{tcount} = trial;
    tcount = tcount + 1;
    tstarts(1) = []; % delete this trial from the list
end

trials = cell2mat(trials);
cndlist = arrayfun(@(x) x.cnd,trials);
cnds = unique(cndlist);

% find trials with REWARD or CORRECT codes
rewarded = arrayfun(@(x) sum(x.codes(:,1) == REWARD)>0,trials);
correct = arrayfun(@(x) sum(x.codes(:,1) == CORRECT)>0,trials);

% use whichever had the greater number of trials
if (sum(rewarded) == 0 & sum(correct) == 0)
    error('No rewarded or correct trials were found');
elseif (sum(rewarded) >= sum(correct))
    goodtrials = rewarded;
else
    goodtrials = correct;
end

% find trials with both a start and an end
started = arrayfun(@(x) sum(x.codes(:,1) == START_TRIAL)>0,trials);
ended = arrayfun(@(x) sum(x.codes(:,1) == END_TRIAL)>0,trials);
completed = started + ended - 1;
completed(find(completed<0)) = 0;

% only include completed trials in the good trials list
goodtrials = boolean(goodtrials .* completed);

EVENTS = cell(size(channels,1),max(cnds),1);
MSGS = cell(max(cnds),1);
DIODE = cell(max(cnds),1);
CODES = cell(max(cnds),1);
ENV = cell(max(cnds),1);

% find all the codes sent between trials and put them in 'params'
preTrial = cell(0);
lastEnd = 0;
params = struct();
for i = 1:length(trials)
    tri = trials(i);
    preCodes = codes(codes(:,2) > lastEnd & codes(:,2) < tri.startTime,1);
    preTrial{i} = [char(preCodes(preCodes >= 256 & preCodes < 512) - 256)'];
    if ~isempty(preCodes)
%         disp(['Trial # ',num2str(i),' preceded by digital codes']);
        %disp(preTrial{i});
    end
    variables = regexp(preTrial{i},';','split');
    for j = 1:length(variables)
        k = strfind(variables{j},'=');
        if k            
            lhs = variables{j}(1:k-1);
            rhs = variables{j}(k+1:end);
            % MATT - should we check here to see if any value has
            % changed and report back if it has?
            try
                eval(['params.' lhs '=[' rhs '];']);
            catch
                eval(['params.' lhs '=''' rhs ''';']);
            end
        end
    end
    % put all the params into the trials struct
    trials(i).env = catstruct(trials(i).env, params);    
    
    lastEnd = tri.endTime;
end



%preTrial = cell(0);
%lastStart = 0;
%trialIndex = 1;
%params = struct();
%for i = 1:length(trials)
%    tri = trials(i);
%    preCodes = codes(codes(:,2) > lastStart & codes(:,2) < tri.startTime,1);
%    if length(preTrial) < trialIndex
%        preTrial{trialIndex} = '';
%    end
%    preTrial{trialIndex} = [preTrial{trialIndex} ...
%                   char(preCodes(preCodes >= 256 & preCodes < 512) - 256)'];
%    variables = regexp(preTrial{trialIndex},';','split');
%    for j = 1:length(variables)
%        k = strfind(variables{j},'=');
%        if k            
%            lhs = variables{j}(1:k-1);
%            rhs = variables{j}(k+1:end);
%            try
%                eval(['params.' lhs '=[' rhs '];']);
%            catch
%                eval(['params.' lhs '=''' rhs ''';']);
%            end
%        end
%    end
%    
%    trials(i).env = catstruct(trials(i).env, params);
%    
%    lastStart = tri.endTime;
%    if goodtrials(i)
%        trialIndex = trialIndex + 1;
%    end
%end

for i = 1:max(cnds)
    theseTrials = trials(cndlist==i & goodtrials);
    
    for j = 1:length(theseTrials)
        if alignFlag
            if alignCode < 0
                
                alignTimeOn = find(theseTrials(j).diode > thresh);
                alignTimesamples=diff(alignTimeOn);
                offtimeind=find(alignTimesamples>1);
%                 disp('FYI: aligning to diode turning off')
                if (isempty(alignTimeOn))
                    error(['No diode flash found - trial ',num2str(j)]);
                end
                %editing this out to align to diode off SBK 02/01/2016
                alignTime = alignTimeOn(1)/diodeSamp;
                alignTimeOff = alignTimeOn(offtimeind+1)/diodeSamp;
                DiodeDuration{i,j}=[alignTime,alignTimeOff,alignTimeOff-alignTime];  
            else
                if (numel(alignCode)==1)
                    alignTime = theseTrials(j).codes(theseTrials(j).codes(:,1) ...
                                                     == alignCode,2);
                elseif (numel(alignCode)>=1)
                    codestr=num2str(theseTrials(j).codes(:,1)');
                    patstr = [];
                    for I=1:length(alignCode)-1
                        patstr = [patstr,num2str(alignCode(I)),' \s '];
                    end
                    patstr = [patstr,num2str(alignCode(I+1))];
                    idx = regexp(codestr,patstr,'start');
                    alignTime = 0;
                end
                if isempty(alignTime)
                    disp(sprintf(['Repeat %i of condition %i does not ' ...
                                  'have align code %i'],j,i, ...
                                 alignCode));
                    alignTime = 0;
                elseif length(alignTime) > 1
                    if ~warnCodeFlag
                        disp(sprintf(['Repeat %i of condition %i has ' ...
                                      '%i occurrences of align code %i ' ...
                                      '- using 1st occurrence'],j,i, ...
                                     length(alignTime),alignCode));
                        disp(['No more warnings of this type will be displayed']);
                        warnCodeFlag = 1;
                    end
                    alignTime = alignTime(1);
                end
            end
        else
            alignTime = 0;
        end

        for k = 1:size(channels,1)
            valid = theseTrials(j).spikes(:,1) == channels(k,1) ...
                        & theseTrials(j).spikes(:,2) == channels(k,2);
                 
            EVENTS{k,i,j} = theseTrials(j).spikes(valid,3) - alignTime;
        end

        CODES{i,j} = theseTrials(j).codes;
        CODES{i,j}(:,2) = CODES{i,j}(:,2) - alignTime;
        if isfield(theseTrials(j),'diode')
            DIODE{i,j} = theseTrials(j).diode;
            theseTrials(j).diodeTime = theseTrials(j).diodeTime - alignTime;
            DIODETIME{i,j} = theseTrials(j).diodeTime;            
        end
        MSGS{i,j} = theseTrials(j).msgs;
        ENV{i,j} = theseTrials(j).env;
    end
end

ex = struct();
ex.EVENTS = EVENTS;
if ~isempty(diode)
    ex.DIODE = DIODE;
    ex.DIODETIME = DIODETIME;
    ex.DiodeDuration=DiodeDuration;
end
ex.MSGS = MSGS;
ex.CODES = CODES;
ex.CHANNELS = channels;
ex.TRIAL_SEQUENCE = cndlist(goodtrials);
ex.REPEATS = hist(ex.TRIAL_SEQUENCE,1:max(cnds));
ex.PRE_TRIAL = preTrial(goodtrials);
ex.ENV = ENV;
