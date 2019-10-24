addpath('preprocess');
addpath(genpath('~/projects/matlab/colorspace'));
FILE_DIR = '~/projects/04-MattSmithV4';
PROJECT_DIR = '~/projects/25-V4py';

HEIGHT = 1001;
WIDTH = 1001;

%% Load the data
file = dir(sprintf('%s/data/booboo/*hues*nev*mat', FILE_DIR));
f = 1;
clearvars -except file f PROJECT_DIR FILE_DIR HEIGHT WIDTH
for f=1:numel(file)
    filename = file(f).name;

    load(sprintf('%s/data/booboo/%s', FILE_DIR, filename));

    [InfoNeuron,SpikeTimes] = getNeurons2(nev);

    % Get spiketimes
    Spikes = [];
    for uu=1:numel(SpikeTimes)
        Spikes(uu).unit = InfoNeuron(uu,1)+0.1*InfoNeuron(uu,2);
        Spikes(uu).times = SpikeTimes{uu};
    end

    % Extract trial onset times
    load(sprintf('%s/data/booboo/%s', FILE_DIR, filename));
    trialonset = find(nev(:,1) == 0 & nev(:,2) == 1);
    trialoffset = find(nev(:,1) == 0 & nev(:,2) == 255);
    [trialonset, trialoffset] = clean_onset_offset_times(trialonset, trialoffset, 1e3*max(vertcat(Spikes(:).times)));
    
    % Extract trial codes
    code_idx = find(nev(:,1) == 0 & nev(:,2) > 255);

    % Extract stimulus onset times
    stimonset = find(nev(:,1) == 0 & nev(:,2) == 10);
    stimoffset = find(nev(:,1) == 0 & nev(:,2) == 40);    
    [stimonset, stimoffset] = clean_onset_offset_times(stimonset, stimoffset, 1e3*max(vertcat(Spikes(:).times)));
    
    % Extract saccades and fixations
    tmp = regexp(filename, '_', 'split');
    stump = '';
    for tt=1:numel(tmp)-1
        stump = [stump, tmp{tt}, '_'];
    end
    stump = [stump, 'eyes.mat'];

    load(sprintf('%s/data/booboo/%s', FILE_DIR, stump));
    eyeh = eyes(1,1:3:end); eyev = eyes(2,1:3:end);
    
    onset = [];
    offset = [];
    for tr=1:min(numel(trialonset), numel(trialoffset))
        % Timestamps of stimuli
        onset = [onset; nev(stimonset(stimonset > trialonset(tr) & stimonset < trialoffset(tr)), 3)];
        offset = [offset; nev(stimoffset(stimoffset > trialonset(tr) & stimoffset < trialoffset(tr)), 3)];    
    end
    
    screenDistance = 36;
    pixPerCM = 27.03;
    
    % Extract hue angle of stimuli
    for tr=1:min(numel(onset), numel(offset))
        this_trial_code_idx = code_idx(code_idx >= trialonset(tr)+1 & code_idx < trialoffset(tr));
        trialmeta(tr).data = char(nev(this_trial_code_idx,2)-256)';
        fprintf('%s\n', trialmeta(tr).data);
    
        % Plain hue
        t_hue = regexp(trialmeta(tr).data, 'suffix=(\w*.\w*);', 'tokens');
        for tmp=1:length(t_hue)
            tt = t_hue{tmp};
            huefile = str2num(tt{1,1}(1:end));
            load(sprintf('%s/stimuli/allMat/hues/v4fv_hp_%d.mat', FILE_DIR, huefile));
            if(~ismember(huefile, [9 10 11]))
              H = colorspace('RGB->Luv', mov{1,1});
              hue(tr,tmp) = atan2(H(1,1,3), H(1,1,2));
            else
              hue(tr,tmp) = NaN;
            end
        end
    
        [r,c] = eyes2rowcol(eyeh(round(1e3*onset(tr)):round(1e3*offset(tr))), ...
                            eyev(round(1e3*onset(tr)):round(1e3*offset(tr))), screenDistance, pixPerCM, HEIGHT, WIDTH);
        row = round(mean(r)); col = round(mean(c));
        
        % Create Eyes data structure
        Eyes(tr).trial = tr;
        Eyes(tr).impath =  'stimuli/M2/Hues';
        Eyes(tr).imname = sprintf('%04d.jpg', huefile);
        Eyes(tr).stim_onset = onset(tr);
        Eyes(tr).stim_offset = offset(tr);
        Eyes(tr).row = row;
        Eyes(tr).col = col;
    end
    
    % Accumulate events
    events.onset = onset;
    events.offset = offset;
    
    % Accumulate features
    features.hue = hue;
    
    save(sprintf('../../matdata/M2/%s', filename), 'Spikes', 'Eyes', 'features', 'events', '-v7.3');
    %pause
end

