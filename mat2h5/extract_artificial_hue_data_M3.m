addpath('preprocess');
addpath(genpath('~/projects/matlab/colorspace'));
FILE_DIR = '~/projects/04-MattSmithV4';
PROJECT_DIR = '~/projects/25-V4py';

HEIGHT = 1001;
WIDTH = 1001;

%% Load the data
file = dir(sprintf('%s/data/freeview_roo2015/*huecircle*nev*mat', FILE_DIR));
f = 1;
clearvars -except file f PROJECT_DIR FILE_DIR HEIGHT WIDTH
for f=9:numel(file)
    filename = file(f).name;

    load(sprintf('%s/data/freeview_roo2015/%s', FILE_DIR, filename));

    [InfoNeuron,SpikeTimes] = getNeurons2(nev);

    % Get spiketimes
    Spikes = [];
    for uu=1:numel(SpikeTimes)
        Spikes(uu).unit = InfoNeuron(uu,1)+0.1*InfoNeuron(uu,2);
        Spikes(uu).times = SpikeTimes{uu};
    end

    % Extract trial onset times
    load(sprintf('%s/data/freeview_roo2015/%s', FILE_DIR, filename));
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

    load(sprintf('%s/data/freeview_roo2015/%s', FILE_DIR, stump));
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
    hue = [];
    count = 1;
    
    for tr=1:min(numel(trialonset), numel(trialoffset))
        this_trial_code_idx = code_idx(code_idx >= trialonset(tr)+1 & code_idx < trialoffset(tr));
        trialmeta(tr).data = char(nev(this_trial_code_idx,2)-256)';
        fprintf('%s\n', trialmeta(tr).data);
    
        % Plain hue
        t_hue = regexp(trialmeta(tr).data, 'color\w=\d+', 'match');
        this_trial_onsets = nev(stimonset(stimonset > trialonset(tr) & stimonset < trialoffset(tr)), 3);
        this_trial_offsets = nev(stimoffset(stimoffset > trialonset(tr) & stimoffset < trialoffset(tr)), 3);
        
        n_stim = numel(this_trial_onsets);
        
        row = [];
        col = [];
        for stim=1:n_stim
            tt = t_hue{3*stim-2}; R = str2num(tt(8:end));
            tt = t_hue{3*stim-1}; G = str2num(tt(8:end));
            tt = t_hue{3*stim}; B = str2num(tt(8:end));
            H = colorspace('RGB->Luv', [R G B]);
            %hue(tr,stim) = round(180/pi*atan2(H(3), H(2))/45)*45;
            hue = [hue; atan2(H(3), H(2))];
            
            hue360 = 180/pi*hue(end);
            if hue360 < 0, hue360 = hue360 + 360; end
            if hue360 == 360, hue360 = 0; end
            hue360 = round(hue360);
                        
            [r,c] = eyes2rowcol(eyeh(round(1e3*this_trial_onsets(stim)):round(1e3*this_trial_offsets(stim))), ...
                            eyev(round(1e3*this_trial_onsets(stim)):round(1e3*this_trial_offsets(stim))), screenDistance, pixPerCM, HEIGHT, WIDTH);
                        
            row = round(mean(r));
            col = round(mean(c));
            
             % Create Eyes data structure
            Eyes(count).trial = tr;
            Eyes(count).impath =  'stimuli/M3/Hues';
            Eyes(count).imname = sprintf('%03d.jpg', hue360);
            Eyes(count).stim_onset = this_trial_onsets(stim);
            Eyes(count).stim_offset = this_trial_offsets(stim);
            Eyes(count).row = row;
            Eyes(count).col = col;
            count = count+1;
        end
    end
    
    % Accumulate events
    events.onset = onset;
    events.offset = offset;
    
    % Accumulate features
    features.hue = hue;
    
    save(sprintf('../../matdata/M3/%s', filename), 'Spikes', 'Eyes', 'features', 'events', '-v7.3');
    %pause
end

