addpath('preprocess');
FILE_DIR = '/media/pavan/Windows7_OS/Work/NUProjects/04-MattSmithV4';
PROJECT_DIR = '/media/pavan/Windows7_OS/Work/NUProjects/25-V4py';

HEIGHT = 1024;
WIDTH = 768;

%% Load the data
file = dir(sprintf('%s/data/booboo/*orig*nev*mat', FILE_DIR));
f = 1;
clearvars -except file f PROJECT_DIR FILE_DIR HEIGHT WIDTH
for f=1:numel(file)
    filename = file(f).name;

    load(sprintf('%s/data/booboo/%s', FILE_DIR, filename));

    [InfoNeuron,SpikeTimes] = getNeurons2(nev);

    % Extract trial onset times
    [onset, offset, stimID] = getTrials3(nev);

    % Extract saccades and fixations
    tmp = regexp(filename, '_', 'split');
    stump = '';
    for tt=1:numel(tmp)-1
        stump = [stump, tmp{tt}, '_'];
    end
    stump = [stump, 'eyes.mat'];

    load(sprintf('%s/data/booboo/%s', FILE_DIR, stump));
    eyeh = eyes(1,1:3:end); eyev = eyes(2,1:3:end);
    EyeData = getSaccades2(onset, offset, eyeh, eyev);

    %% Get behavior variables for each trial
    
    %-------------------
    % Go trial-by-trial
    %-------------------
    count = 0;
    for tr=1:length(onset)
        fprintf('Trial #%03d\n', tr);

        % Trial number
        trial = tr;
        
        % Image path
        impath = 'stimuli/M2/Orig';
        
        % Image name
        imageID = stimID(tr);
        imname = sprintf('%04d.jpg', imageID);
        
        %----------------------
        % Go saccade-by-saccade
        %----------------------
        for s=1:numel(EyeData(tr).sacend)-1
            
            % Fixation number
            fixation = s;
            
            % Row and column of fixation
            fixstart = EyeData(tr).sacend(s);
            fixend = EyeData(tr).sacstart(s+1);
            row = round(nanmean(EyeData(tr).pixeyev(fixstart:fixend)));
            col = round(nanmean(EyeData(tr).pixeyeh(fixstart:fixend)));

            % Row and column at fixation onset
            fix_onset_row = round(EyeData(tr).pixeyev(EyeData(tr).sacend(s)));
            fix_onset_col = round(EyeData(tr).pixeyeh(EyeData(tr).sacend(s)));
            
            % Row and column at fixation offset
            fix_offset_row = round(EyeData(tr).pixeyev(EyeData(tr).sacstart(s+1)));
            fix_offset_col = round(EyeData(tr).pixeyeh(EyeData(tr).sacstart(s+1)));
            
            % Fixation onset time
            fix_onset = 1e-3*(onset(tr) + EyeData(tr).sacend(s));
            
            % Fixation offset time
            fix_offset = 1e-3*(onset(tr) + EyeData(tr).sacstart(s+1));
            
            % Incoming saccade duration
            in_sac_dur = 1e-3*(EyeData(tr).sacend(s)-EyeData(tr).sacstart(s));
            
            % Incoming saccade peak velocity
            in_sac_pkvel = EyeData(tr).pkvel(s);
            
            % Incoming saccade blink or not?
            in_sac_blink = EyeData(tr).blink(s);
            
            % Outgoing saccade duration
            out_sac_dur = 1e-3*(EyeData(tr).sacend(s+1)-EyeData(tr).sacstart(s+1));

            % Outgoing saccade peak velocity
            out_sac_pkvel = EyeData(tr).pkvel(s+1);
            
            % Outgoing saccade blink or not?
            out_sac_blink = EyeData(tr).blink(s+1);
            
            % Is it a bad fixation
            badfix = EyeData(tr).badfix(s);
            
            % Put everything into a struct
            count=count+1;
            Eyes(count).trial = trial;
            Eyes(count).impath = impath;
            Eyes(count).imname = imname;
            Eyes(count).fixation = fixation;
            Eyes(count).in_sac_dur = in_sac_dur;
            Eyes(count).in_sac_pkvel = in_sac_pkvel;
            Eyes(count).in_sac_blink = in_sac_blink;
            Eyes(count).out_sac_dur = out_sac_dur;
            Eyes(count).out_sac_pkvel = out_sac_pkvel;
            Eyes(count).out_sac_blink = out_sac_blink;
            Eyes(count).fix_onset = fix_onset;
            Eyes(count).fix_offset = fix_offset;
            Eyes(count).row = row;
            Eyes(count).col = col;
            Eyes(count).fix_onset_row = fix_onset_row;
            Eyes(count).fix_onset_col = fix_onset_col;
            Eyes(count).fix_offset_row = fix_offset_row;
            Eyes(count).fix_offset_col = fix_offset_col;
            Eyes(count).badfix = badfix;
            
        end
    end

    % Compute spike trains
    Spikes = [];
    for uu=1:numel(SpikeTimes)
        Spikes(uu).unit = InfoNeuron(uu,1)+0.1*InfoNeuron(uu,2);
        Spikes(uu).times = SpikeTimes{uu};
    end
    
    save(sprintf('../../matdata/M2/%s', filename), 'Eyes', 'Spikes', '-v7.3');
    %pause
end

