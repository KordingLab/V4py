addpath('../preprocess');
addpath('../glmcode-dev');
addpath(genpath('../../../matlab/CircStat2012a/'))
addpath(genpath('../../../matlab/colorspace/'));

% Define some constants
params.PATCHSIZE = 301;    % This currently corresponds to 12 x 12 deg roughly.
params.HEIGHT = 768; params.WIDTH = 1024;
params.BINSIZE = 10;
params.PAD = 50;    %(500 ms)
% Toggle between the conversion method 'exact' or 'approx'
% 'exact' is slower but more precise
params.LUVMETHOD = 'approx';

% Define temporal basis functions
[time, vBasis] = rcosbasis(1:30, [5 8 15], [3 4 5]);
params.time = time;
params.vBasis = vBasis;

[~, mBasis] = rcosbasis(1:20, [5 8], [3 5]);
params.mBasis = mBasis;

% Toggle between stimulus and shuffled control covariates
% Set this to 1 if you want to compute covariates for shuffled controls
% otherwise 0 by default
params.CTRL = 1;


%% Load the data
file = dir('../../data/freeview_roo2015/*FreeViewPic*nev*mat');
f = 1;
clearvars -except file params f
for f=1:numel(file)
filename = file(f).name;

load(sprintf('../../data/freeview_roo2015/%s', filename));

[InfoNeuron,SpikeTimes] = getNeurons2(nev);

% Extract trial onset times
[onset, offset, stimID, condID] = getTrials4(nev);

% Extract saccades and fixations
tmp = regexp(filename, '_', 'split');
stump = '';
for tt=1:numel(tmp)-1
    stump = [stump, tmp{tt}, '_'];
end
stump = [stump, 'eyes.mat'];

load(sprintf('../../data/freeview_roo2015/%s', stump));
eyeh = eyes(1,1:3:end); eyev = eyes(2,1:3:end);
EyeData = getSaccades2(onset, offset, eyeh, eyev);

%% Compute covariates

PATCHSIZE = params.PATCHSIZE;
p = PATCHSIZE;
HEIGHT = params.HEIGHT;
WIDTH = params.WIDTH;
BINSIZE = params.BINSIZE;
LUVMETHOD = params.LUVMETHOD;
CTRL = params.CTRL;
%RF = params.RF;
%RFsum = params.RFsum;
PAD = params.PAD;

vBasis = params.vBasis;
mBasis = params.mBasis;

X = [];
XL = [];
XHv = [];

%-----------------------
% Covariate computation
%-----------------------
Covs.H.mean = []; Covs.H.mode = []; Covs.H.std = [];
Covs.S.mean = []; Covs.S.mode = []; Covs.S.std = [];
Covs.L.mean = []; Covs.L.mode = []; Covs.L.std = [];

tr_idx = [];
sac_idx = [];

%-------------------
% Go trial-by-trial
%-------------------
fix_idx = 1;
for tr=1:length(onset)
    fprintf('Trial #%03d\n', tr);

    % Read in the trial-specific stimulus
    if(CTRL == 0), imageID = stimID(tr);
    else imageID = randi(300,1,1);
    end
    if(strcmp(condID{tr}, 'eqlum3'))
        imname = sprintf('../../stimuli/Scenes/EqLum3/%04d.jpg', imageID);
    elseif(strcmp(condID{tr}, 'orig'))
        imname = sprintf('../../stimuli/Scenes/Orig/%04d.jpg', imageID);
    elseif(strcmp(condID{tr}, 'histeq'))
       imname = sprintf('../../stimuli/Scenes/Histeq/%04d.jpg', imageID);
    elseif(strcmp(condID{tr}, 'randlum3'))
       imname = sprintf('../../stimuli/Scenes/RandLum3/%04d.jpg', imageID);
    elseif(strcmp(condID{tr}, 'Boston1'))
       imname = sprintf('../../NewStimuli/LabelMe/Boston/set1/%04d.jpg', imageID);
    elseif(strcmp(condID{tr}, 'Boston2'))
       imname = sprintf('../../NewStimuli/LabelMe/Boston/set2/%04d.jpg', imageID);
    elseif(strcmp(condID{tr}, 'Other'))
       imname = sprintf('../../NewStimuli/LabelMe/Other/%04d.jpg', imageID);
    end
    I = imread(imname);

    % Reject the trial if the image size is bad
    if(size(I,1) ~= HEIGHT || size(I,2) ~= WIDTH), continue; end

    % Pad the image with gray surround area
    I_pad = 128*ones(HEIGHT*2, WIDTH*2, 3);
    I_pad(HEIGHT/2:HEIGHT/2+size(I,1)-1, WIDTH/2:WIDTH/2+size(I,2)-1,:) = I;

    % Compute image features
    %------------------------
    % Convert to HSI space
    H_pad = colorspace('RGB->HSL', I_pad/255); % convert this to hue values (in angles; radians)

    % Compute orientation statistics for the whole image
    I_gray = rgb2gray(double(I_pad)/255);

    %----------------------
    % Go saccade-by-saccade
    %----------------------
    for s=1:numel(EyeData(tr).sacend)-1
        fprintf('\t Saccade #%03d\n', s);
        row = round(HEIGHT-median(EyeData(tr).pixeyev(EyeData(tr).sacend(s):EyeData(tr).sacstart(s+1))));
        col = round(median(EyeData(tr).pixeyeh(EyeData(tr).sacend(s):EyeData(tr).sacstart(s+1))));

        if(row <= p/2 || col <= p/2 || row >= HEIGHT-p/2 || col >= WIDTH-p/2 || isnan(row) || isnan(col))
            continue;
        end

        %Extract the HSI patch at current saccade (H)
        H = double(H_pad(HEIGHT/2+row-(p-1)/2:HEIGHT/2+row+(p-1)/2, WIDTH/2+col-(p-1)/2:WIDTH/2+col+(p-1)/2, :));

        % Block average the patches
        %figure; imagesc(H(:,:,1)); colormap(gray); colorbar;
        %figure; imagesc(Hv, [0 90]); colormap(gray); colorbar; pause;
        [Hm, Hv, Hmax, Hdist] = gridimage(H(:,:,1), 4, 4, 'circ');
        [Sm, Sv, Smax, Sdist] = gridimage(H(:,:,2), 4, 4, 'notcirc');
        [Lm, Lv, Lmax, Ldist] = gridimage(H(:,:,3), 4, 4, 'notcirc');

        % Prepare the covariates for this saccade
        u = cos(Hm*pi/180); v = sin(Hm*pi/180);
        thisSac.H.mean = [u(:); v(:)]';
        u = cos(Hmax*pi/180); v = sin(Hmax*pi/180);
        thisSac.H.mode = [u(:); v(:)]';
        thisSac.H.std = Hv(:)';

        thisSac.L.mean = Lm(:)';
        thisSac.L.mode = Lmax(:)';
        thisSac.L.std = Lv(:)';

        thisSac.S.mean = Sm(:)';
        thisSac.S.mode = Smax(:)';
        thisSac.S.std = Sv(:)';

        % Accumulate
        Covs.H.mean = [Covs.H.mean; thisSac.H.mean];
        Covs.H.mode = [Covs.H.mode; thisSac.H.mode];
        Covs.H.std = [Covs.H.std; thisSac.H.std];

        Covs.S.mean = [Covs.S.mean; thisSac.S.mean];
        Covs.S.mode = [Covs.S.mode; thisSac.S.mode];
        Covs.S.std = [Covs.S.std; thisSac.S.std];

        Covs.L.mean = [Covs.L.mean; thisSac.L.mean];
        Covs.L.mode = [Covs.L.mode; thisSac.L.mode];
        Covs.L.std = [Covs.L.std; thisSac.L.std];

        tr_idx = [tr_idx; tr];
        sac_idx = [sac_idx; s];
        image_name{fix_idx} =  imname;
        coords(fix_idx,:) = [row, col];
        fix_idx = fix_idx+1;

        %-----------------------------------------------
%                 % Debugging
%                   figure; imshow(I_pad/255);
%                   hold on;
%                   line((WIDTH/2+col-(p-1)/2)*ones(1,p), HEIGHT/2+row-(p-1)/2:HEIGHT/2+row+(p-1)/2);
%                   line((WIDTH/2+col+(p-1)/2)*ones(1,p), HEIGHT/2+row-(p-1)/2:HEIGHT/2+row+(p-1)/2);
%                   line(WIDTH/2+col-(p-1)/2:WIDTH/2+col+(p-1)/2, (HEIGHT/2+row-(p-1)/2)*ones(1,p));
%                   line(WIDTH/2+col-(p-1)/2:WIDTH/2+col+(p-1)/2, (HEIGHT/2+row+(p-1)/2)*ones(1,p));
%
%                   figure;
%                   subplot(4,2,1); imagesc(I_gray); colormap(gray);
%                   %subplot(4,2,2); imagesc(RF(1).mask);
%                   subplot(4,2,3); imagesc(cos(2*OR), [-1 1]);
%                   subplot(4,2,4); imagesc(sin(2*OR), [-1 1]);
%                   subplot(4,2,5); imagesc(u, [-1 1]);
%                   subplot(4,2,6); imagesc(v, [-1 1]);
%                   subplot(4,2,7); imagesc(L, [0 1]);
%                   subplot(4,2,8); imagesc(Sat, [0 1]);
%                   close all
        %-----------------------------------------------
    end
end

%% % Compute spike trains
addpath('../glmcode-dev/');
LAG = 10;   % ms
LEAD = 10;   % ms
WINDOW = 100;   % ms
BINSIZE = params.BINSIZE;   % ms
HEIGHT = params.HEIGHT;
WIDTH = params.WIDTH;
p = params.PATCHSIZE;

Y = zeros(size(Covs.H.mean,1), numel(SpikeTimes));
FixDur = [];

for neuron = 1:numel(SpikeTimes);
    Yneuron = [];
    for tr=1:length(onset)
        spikes = histc(SpikeTimes{neuron}, [onset(tr)/1000:BINSIZE/1000:offset(tr)/1000+5]);
        for s=1:numel(EyeData(tr).sacend)-1
            row = round(HEIGHT-median(EyeData(tr).pixeyev(EyeData(tr).sacend(s):EyeData(tr).sacstart(s+1))));
            col = round(median(EyeData(tr).pixeyeh(EyeData(tr).sacend(s):EyeData(tr).sacstart(s+1))));

            if(row <= p/2 || col <= p/2 || row >= HEIGHT-p/2 || col >= WIDTH-p/2 || isnan(row) || isnan(col))
                continue;
            end

            % Get spike rate for this saccade
            spikecount = sum(spikes(round((EyeData(tr).sacend(s)+LAG)/BINSIZE):round((EyeData(tr).sacstart(s+1)-LEAD)/BINSIZE)));
            duration = ((EyeData(tr).sacstart(s+1)-LEAD) - (EyeData(tr).sacend(s)+LAG))/1000;
            thisSac.Y = spikecount/duration;    % mean f.r. over entire fixation

            %START = round((EyeData(tr).sacend(s))/BINSIZE);
            %FIN = min(round((EyeData(tr).sacstart(s+1))/BINSIZE), round(START+WINDOW/BINSIZE));
            %thisSac.Y = max(spikes(START:FIN))/BINSIZE*1000;    % peak f.r. over 50-150 ms post-fixation

            % Get fixation duration
            if(neuron==1)
                thisSac.FixDur = (EyeData(tr).sacstart(s+1) - EyeData(tr).sacend(s));
                FixDur = [FixDur; thisSac.FixDur];
            end

            % Accumulate
            Yneuron = [Yneuron; thisSac.Y];
        end
    end
    Y(:,neuron) = Yneuron;
end

save(sprintf('../../data/freeview_roo2015/precomputedCovs2/Shuffle_%s', filename), 'Covs', 'tr_idx', 'sac_idx', 'Y', 'FixDur', 'image_name', 'coords');
%pause
end
break

%% Quick fix files
% file = dir('../../data/freeview_roo2015/*FreeViewPic*nev*mat');
%
% for f=1:numel(file)
%     filename = file(f).name;
%     load(sprintf('../../data/freeview_roo2015/precomputedCovs2/%s', filename));
%
%     % Reshape
%     Covs.L.mean = Covs.L.mean'; %reshape(Covs.L.mean, [16 size(Covs.L.mean,1)/16]);
%     Covs.L.mode = Covs.L.mode'; %reshape(Covs.L.mode, [16 size(Covs.L.mode,1)/16]);
%     Covs.S.mean = Covs.S.mean'; %reshape(Covs.S.mean, [16 size(Covs.S.mean,1)/16]);
%     Covs.S.mode = Covs.S.mode'; %reshape(Covs.S.mode, [16 size(Covs.S.mode,1)/16]);
%
%     save(sprintf('../../data/freeview_roo2015/precomputedCovs2/%s', filename), 'Covs', 'tr_idx', 'sac_idx', 'Y', 'FixDur');
% end

%% Setup cross-validation
rng(2);
addpath(genpath('../../../matlab/glmnet_matlab'));
% Load precomputed covariates
load(sprintf('../../data/freeview_roo2015/precomputedCovs2/Shuffle_%s', filename));

% Remove some artifacts
[i,j] = find(Y < 0);
Covs.H.mode(unique(i),:) = [];
Covs.H.mean(unique(i),:) = [];
Covs.H.std(unique(i),:) = [];

Covs.L.mean(unique(i),:) = [];
Covs.S.mean(unique(i),:) = [];

Y(unique(i),:) = [];
tr_idx(unique(i),:) = [];
FixDur(unique(i),:) = [];

% Select covariate of interest
X = Covs.H.mean;

% If shuffle control
%X = shuffle(X);

%-----------------------
% Subselect fixations
%-----------------------
for tr=1:numel(condID), origID(tr) = strcmp(condID{tr}, 'orig'); end
for tr=1:numel(condID), eqlumID(tr) = strcmp(condID{tr}, 'eqlum3'); end
orig_idx = ismember(tr_idx, find(origID == 1));
eqlum_idx = ismember(tr_idx, find(eqlumID == 1));

% Select fixations that are relatively homogeneous
homogeneous_idx = (max(Covs.H.std, [], 2) < 60);

% Select fixations within a reasonable range of durations (80-600 ms)
fixdur_idx = (FixDur > 80 & FixDur < 600);

% Select fixations within a reasonable range of luminance & saturation (0.2-0.8)
lum_idx = (min(Covs.L.mean, [], 2) > 0.2 & max(Covs.L.mean, [], 2) < 0.8);
sat_idx = (min(Covs.S.mean, [], 2) > 0.2 & max(Covs.S.mean, [], 2) < 0.8);

% Setup cross validation
NCV = 10;
trial_CV = get_cv_inds(tr_idx, NCV);

%% Fit hue tuning
Nboot = 100;

%SPACESELECT = [11 12 15 16];   % bottom right
%SPACESELECT = [1 2 5 6];       % top left
%SPACESELECT = [9 10 13 14];    % top right
%SPACESELECT = [3 4 7 8];       % bottom left
%SPACESELECT = [5];
%SPACESELECT = [1:16];          % all
%SPACESELECT = [SPACESELECT, 16+SPACESELECT];

opts = glmnetSet;
opts.alpha = 0.1; opts.lambda = 0.1;

for neuron = 1:numel(SpikeTimes)
    tic
    fprintf('Neuron: %02d ...', neuron);
    for s=1:size(X,2)/2
        SPACESELECT = [s, 16+s];
        %beta(s,neuron).b(:,1) = glmfit(X(r,SPACESELECT), Y(r,neuron), 'poisson');
        %beta(s,neuron).b(:,2) = glmfit(X(t,SPACESELECT), Y(t,neuron), 'poisson');

        if(nanmean(Y(:,neuron)) > 0.2)  % Exclude low firing neurons
            % Fit each fold to get pseudo-R2s
            for cv=1:NCV
                r = ismember(tr_idx, trial_CV(cv).r);
                t = ismember(tr_idx, trial_CV(cv).t);

                % Discard certain fixations
                r = r & fixdur_idx;% & homogeneous_idx & lum_idx;
                t = t & fixdur_idx;% & homogeneous_idx & lum_idx;

                %fit = glmnet(X(r,SPACESELECT), Y(r,neuron), 'poisson', opts);
                %beta(s,neuron).b0(:,cv) = [fit.a0; fit.beta];
                beta(s,neuron).b0(:,cv) = glmfit(X(r,SPACESELECT), Y(r,neuron), 'poisson')';

                YY(cv).r = Y(r,neuron);
                YY(cv).t = Y(t,neuron);
                YY(cv).t_hat = exp([ones(sum(t),1), X(t,SPACESELECT)]*beta(s,neuron).b0(:,cv));
            end

            % Bootstrap to get confidence bounds on parameters
            %beta(s,neuron).b(:,1:Nboot) = bootstrp(Nboot,@glmnet_boot, X(fixdur_idx,SPACESELECT), Y(fixdur_idx,neuron))';
            beta(s,neuron).b(:,1:Nboot) = bootstrp(Nboot,@glmfit_boot, X(fixdur_idx,SPACESELECT), Y(fixdur_idx,neuron))';

        else
            beta(s,neuron).b0 = zeros(3,NCV);
            beta(s,neuron).b = zeros(3,Nboot);
        end

        % Get pseudo-R2s
        YYr = []; YYt = []; YYt_hat = [];
        for cv=1:NCV
            YYr = [YYr; YY(cv).r];
            YYt = [YYt; YY(cv).t];
            YYt_hat = [YYt_hat; YY(cv).t_hat];
            R2t(s,neuron).Hue.CI(cv,:) = prctile(bootstrp(Nboot, @compute_pseudo_R2, YY(cv).t, YY(cv).t_hat, mean(YY(cv).r)), [2.5 5 50 95 97.5]);
        end
        R2t(s,neuron).Hue.AllCI = prctile(bootstrp(Nboot, @compute_pseudo_R2, YYt, YYt_hat, mean(YYr)), [2.5 5 50 95 97.5]);
    end
    tm = toc;
    fprintf(' %4.2f s [done]\n', tm);
end

save(sprintf('../../data/freeview_roo2015/Natural_Hue_Fits2/Shuffle_glmfit_%s', filename), 'beta', 'R2t', 'InfoNeuron');
%end
break

%% Visualize hue tuning
load(sprintf('../../data/freeview_roo2015/Natural_Hue_Fits2/glmfit_%s', filename));
NCV = 10;
clear RF_hue H_hat H_hat_RGB
for neuron=1:numel(SpikeTimes)
    R2t_neuron = [];
    for s=1:size(R2t,1)
        R2t_neuron_lowCI(s) = R2t(s,neuron).Hue.AllCI(2);
    end
    if(any(R2t_neuron_lowCI > 0))

    figure;
    cv = 1;
    for i=1:4
        for j=1:4
            %fprintf('(%d,%d): %d %d\n', j,i,1 + 4 + 2*(i-1) + j, 1 + 2*(i-1) + j);
            RF(cv).hue(j,i) = atan2(beta(4*(i-1)+j, neuron).b0(3,cv), beta(4*(i-1)+j, neuron).b0(2,cv));
        end
    end
    H_hat(cv).hue(:,:,1) = ones(size(RF(cv).hue)); H_hat(cv).hue(:,:,2) = cos(RF(cv).hue); H_hat(cv).hue(:,:,3) = sin(RF(cv).hue);
    H_hat(cv).RGB = 30*colorspace('Luv->RGB', H_hat(cv).hue);

    % Preferred angle plot
    for i=1:4
        for j=1:4
            if(R2t_neuron_lowCI(4*(i-1)+j) > 0)
            %fprintf('(%d,%d): %d %d\n', i,2-j+1,j,i);
            subplot(2, 1, 1); axis([0 5 0 5]); hold on; plot(i,4-j+1, 'o', 'MarkerSize', 20, 'MarkerEdgeColor', H_hat(cv).RGB(j,i,:), 'MarkerFaceColor', H_hat(cv).RGB(j,i,:));
            end
        end
    end
    title(sprintf('Neuron %03d: Hue tuning\n', neuron));

    % Pseudo-R2 plot
    hold on; subplot(2, 1, 2); plot(0:size(R2t,1)+1, zeros(size(R2t,1)+2, 1), 'k--');
    for s=1:size(R2t,1)
        if(R2t_neuron_lowCI(s) > 0)
            subplot(2, 1, 2); hold on; errorbar(s, R2t(s,neuron).Hue.AllCI(3), R2t(s,neuron).Hue.AllCI(3)-R2t(s,neuron).Hue.AllCI(2), R2t(s,neuron).Hue.AllCI(4)-R2t(s,neuron).Hue.AllCI(3), 'ro', 'MarkerFaceColor', [1 0 0]);
        else
            subplot(2, 1, 2); hold on; errorbar(s, R2t(s,neuron).Hue.AllCI(3), R2t(s,neuron).Hue.AllCI(3)-R2t(s,neuron).Hue.AllCI(2), R2t(s,neuron).Hue.AllCI(4)-R2t(s,neuron).Hue.AllCI(3), 'ko', 'MarkerFaceColor', [0 0 0]);
        end
    end
    title('Hue tuning: pseudo-R2s');
    end
end

%% Population-level statistics
hh = []; mm = []; ss = [];
for cv=1:size(beta(1,1).b,2)
for s=1:size(R2t,1)
    for neuron=1:size(R2t,2)
        if(R2t(s,neuron).Hue.AllCI(2) > 0)
            hh(s,neuron,cv) = atan2(beta(s,neuron).b(3,cv), beta(s,neuron).b(2,cv));
        else
            hh(s,neuron,cv) = NaN;
        end
    end
end
end

figure;
subplot(211); hist(180/pi*hh(:), 25);
xlim([-200 200]);

subplot(212);
for neuron=1:size(R2t,2)
    for s=1:size(R2t,1)
        R2t_neuron_lowCI(s) = R2t(s,neuron).Hue.AllCI(2);
    end
    if(any(R2t_neuron_lowCI > 0))
        fprintf('Neuron %d\n', neuron);
        for s=1:size(R2t,1)
            hhh = squeeze(hh(s,neuron,:));
            tmp1 = hhh(~isnan(hhh));
            if(~isempty(tmp1))
                mm(s) = circ_median(tmp1);
                ss(s,:) = circ_CI(tmp1, 90);
            else
                mm(s) = NaN;
                ss(s,:) = [NaN, NaN];
            end
        end

        mmm = 180/pi*circ_median(mm(~isnan(mm))');
        sss = 180/pi*[circ_median(ss(~isnan(ss(:,1)),1)), circ_median(ss(~isnan(ss(:,2)),2))];
        %ss = 180/pi*circ_confmean(tmp3, 0.05);
        hold on; errorbar(neuron, mmm, mmm-sss(1), sss(2)-mmm, 'ks',...
                 'MarkerFaceColor', 30*colorspace('Luv->RGB', [1 cos(mmm/180*pi) sin(mmm/180*pi)]),...
                 'MarkerEdgeColor', 30*colorspace('Luv->RGB', [1 cos(mmm/180*pi) sin(mmm/180*pi)]));
    end
end
ylim([-250 250]);

%% Compare with artificial RFs

% Load the previously fitted artificial hue tuning curves
tmp = regexp(filename, '_', 'split');
art_file = dir(sprintf('../../data/freeview_roo2015/Artificial_Hue_Fits/*%s*', [tmp{2}]));
load(sprintf('../../data/freeview_roo2015/Artificial_Hue_Fits/%s', art_file(1).name));

% Match neuron indices between artificial and natural
IN = InfoNeuron(:,1) + InfoNeuron(:,2)/10;
IN_Art = InfoNeuron_Artificial(:,1) + InfoNeuron_Artificial(:,2)/10;

for n=1:numel(IN)
    tmp = find(IN_Art == IN(n));
    if(isempty(tmp))
        IN_idx(n) = NaN;
    else
        IN_idx(n) = tmp(1);
    end
end

% Plot the artificial tuning curves overlaid on natural

subplot(212);
for neuron=1:numel(IN)
    for s=1:size(R2t,1)
        R2t_neuron_lowCI(s) = R2t(s,neuron).Hue.AllCI(2);
    end
    %if(any(R2t_neuron_lowCI > 0))
    if(~isnan(IN_idx(neuron)) && R2t_art(IN_idx(neuron)).AllCI(2) > 0)
    hh_art = Hue_hat_artificial(IN_idx(neuron)).hue;
    mm = 180/pi*circ_mean(hh_art'); ss = 180/pi*circ_CI(hh_art', 95);
    hold on; errorbar(neuron, mm, mm-ss(1), ss(2)-mm, 'ko',...
             'MarkerFaceColor', 30*colorspace('Luv->RGB', [1 cos(mm/180*pi) sin(mm/180*pi)]),...
             'MarkerEdgeColor', 30*colorspace('Luv->RGB', [1 cos(mm/180*pi) sin(mm/180*pi)]));
    end
    %end
end
