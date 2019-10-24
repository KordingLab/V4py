function EyeData = getSaccades(onset, offset, eyeh, eyev)

close all

% Voltage to degrees
eyeh = medfilt1(double(eyeh), 5)/575;
eyev = medfilt1(double(eyev), 5)/575;

% Degrees to pixels
%hfac = 1024/41.7;
%vfac = 768/33.7;
%pixeyeh = pixeyeh*hfac + 512;
%pixeyev = pixeyev*vfac + 384;

screenDistance = 43;
pixPerCM = 26.75;
pixeyev = 768/2 - pixPerCM*screenDistance*tan(eyev*pi/180);
pixeyeh = 1024/2 + pixPerCM*screenDistance*tan(eyeh*pi/180);

% Detect saccades
plotstep = 10;

% Design saccade filter
b = fir1(100, 25/500);

% Set some thresholds
MINPEAK = 5; %pixels/ms
BLINKTHRESH = 1000; %pixels/ms
STOPTHRESH = 1; % pixels/ms
PATHDEVIATION = 20; % pixels

for tr=1:length(onset)
TRLEN = offset(tr)-onset(tr)+1;
EyeData(tr).goodtrial = 0;
if(TRLEN > 1000)
    EyeData(tr).goodtrial = 1;
    fprintf('Trial #%03d\n', tr);
    if(onset(tr) ~= 0 && offset(tr) ~= 0)
        hvel = diff(pixeyeh(onset(tr):offset(tr)));
        vvel = diff(pixeyev(onset(tr):offset(tr)));
        hvel = filtfilt(b,1,hvel);
        vvel = filtfilt(b,1,vvel);
        vel = hvel.^2+vvel.^2;
    end
    
    % Scale the plot to show out-of-image deflections
    %%%figure(2); hold on;
    %%%axis([-400 1424 -400 1168])
    
    % Display the stimulus image
    %%%I = imread(sprintf('../data/scenes/scenes/morescene%04d.jpg', 1000+tr-3));
    %%%imagesc(flipdim(I,1));
    %%%colormap('gray'); 
    
    % Animate the scan path
%     for t=onset(tr):plotstep:offset(tr)
%       plot(pixeyeh(t), pixeyev(t), 'kx', 'Markersize', 10);
%       pause(0.0001);
%     end
    
    % Detect peaks
    [pks, pktimes] = findpeaks(vel, 'minpeakheight', MINPEAK);
    
    % Detect blinks
%     if(max(pks) > BLINKTHRESH)
%       blinkidxs = find(pks > 1.2*std(pks));
%       % Mark peaks adjacent to blink peaks for deletion
%       adjidxs = [];
%       for bk = 1:length(blinkidxs)
%         if(blinkidxs(bk) > 1)
%           if(isempty(find(pktimes == blinkidxs(bk)-1, 1)) && pktimes(blinkidxs(bk)) - pktimes(blinkidxs(bk)-1) < 100)
%             adjidxs = [adjidxs, blinkidxs(bk)-1];
%           end
%         end
%         if(blinkidxs(bk) < length(pks))
%           if(isempty(find(pktimes == blinkidxs(bk)+1, 1)) && pktimes(blinkidxs(bk)+1) - pktimes(blinkidxs(bk)) < 100)
%             adjidxs = [adjidxs, blinkidxs(bk)+1];
%           end
%         end
%       end
%       blinkidxs = [blinkidxs, adjidxs];
%     
%       % Discard blinks
%       pks(blinkidxs) = [];
%       pktimes(blinkidxs) = [];
%     end
    
    % Detect saccades
    for p=1:length(pktimes)
      % Store peak velocity
      EyeData(tr).pkvel(p) = pks(p);
      
      % Label blinks
      EyeData(tr).blink(p) = (pks(p) > BLINKTHRESH);
      
      % Store peak velocity timestamp
      EyeData(tr).sacpeak(p) = pktimes(p);
      
      % Detect and store saccade start timestamp
      temp = find(vel(max(pktimes(p)-100,1):pktimes(p)) < STOPTHRESH);
      if(~isempty(temp))
        EyeData(tr).sacstart(p) = max(pktimes(p) - 100 + temp(end), 1);
      else
        EyeData(tr).sacstart(p) = max(EyeData(tr).sacpeak(p) - 40, 1);
      end
      
      % Detect and store saccade end timestamp
      temp = find(vel(pktimes(p):min(pktimes(p)+100,TRLEN-1)) < STOPTHRESH);
      if(~isempty(temp))
        EyeData(tr).sacend(p) = pktimes(p) + temp(1);
      else
        EyeData(tr).sacend(p) = min(EyeData(tr).sacpeak(p) + 40, TRLEN-1);
      end 
    end
    
    % Exclude duplicates
    toDel = [];
    for p=1:length(EyeData(tr).sacend)
        tmp = find(EyeData(tr).sacstart == EyeData(tr).sacstart(p));
        if(numel(tmp) > 1)
            toDel = [toDel, tmp(1:end-1)];
        end
        tmp = find(EyeData(tr).sacend == EyeData(tr).sacend(p));
        if(numel(tmp) > 1)
            toDel = [toDel, tmp(1:end-1)];
        end
    end
    if(numel(EyeData(tr).sacend) > 0)
        EyeData(tr).blink(toDel) = [];
        EyeData(tr).pkvel(toDel) = [];
        EyeData(tr).sacstart(toDel) = [];
        EyeData(tr).sacpeak(toDel) = [];
        EyeData(tr).sacend(toDel) = [];
    end
    
    % Merge peaks belonging to the same saccade
    tmp = []; sacstarttoDel = []; sacendtoDel = [];
    for p=1:length(EyeData(tr).sacend)-1
        tmp = min(vel(EyeData(tr).sacend(p):EyeData(tr).sacstart(p+1)));
        if(tmp > STOPTHRESH)
            sacstarttoDel = [sacstarttoDel, p+1];
            sacendtoDel = [sacendtoDel, p];
        end
    end
    if(numel(EyeData(tr).sacend) > 0)
        EyeData(tr).sacstart(sacstarttoDel) = [];
        EyeData(tr).sacpeak(sacstarttoDel) = [];
        EyeData(tr).blink(sacstarttoDel) = [];
        EyeData(tr).pkvel(sacstarttoDel) = [];
        EyeData(tr).sacend(sacendtoDel) = [];
    end
    
    % Store kinematics
    EyeData(tr).hvel = hvel;
    EyeData(tr).vvel = vvel;
    EyeData(tr).vel = vel;
    EyeData(tr).pixeyeh = pixeyeh(onset(tr):offset(tr));
    EyeData(tr).pixeyev = pixeyev(onset(tr):offset(tr));
    
    % Mark bad fixations 
    for p=1:length(EyeData(tr).sacend)-1
        fixstart = EyeData(tr).sacend(p);
        fixend = EyeData(tr).sacstart(p+1);
        h = EyeData(tr).pixeyeh(fixstart:fixend);
        v = 768-EyeData(tr).pixeyev(fixstart:fixend);
        
        hm=nanmean(h);
        vm=nanmean(v);
        if(mean(abs(hm-h)) > PATHDEVIATION || mean(abs(vm-v)) > PATHDEVIATION || isnan(hm) || isnan(vm))
            EyeData(tr).badfix(p) = 1;
        else
            EyeData(tr).badfix(p) = 0;
        end
    end
    EyeData(tr).badfix(p+1) = 0;
    
%     % Plot the xy gaze coordinates within the trial
%     figure(200);  cla; hold on;
%     plot(pixeyeh(onset(tr):offset(tr)));
%     plot(pixeyev(onset(tr):offset(tr)), 'r');
%     plot(vel, 'k');
%     
%     % Animate the detected fixations
%     % Plot the scan path
%     figure(100); cla; hold on;
%     plot(pixeyeh(onset(tr):offset(tr)), 768-pixeyev(onset(tr):offset(tr)));
%     for p=1:length(EyeData(tr).sacend)-1
%         
%         % Visualize each fixation
%         fixstart = EyeData(tr).sacend(p);
%         fixend = EyeData(tr).sacstart(p+1);
%         h = EyeData(tr).pixeyeh(fixstart:fixend);
%         v = 768-EyeData(tr).pixeyev(fixstart:fixend);
%         hm=nanmean(h);
%         vm=nanmean(v);
%         mean(abs(hm-h)), mean(abs(vm-v))
%         figure(100); hold on;
%         plot(hm, vm, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
%         
%         
%         % Visualize saccade onsets, peaks and offsets
%         figure(200); hold on;
%         plot(EyeData(tr).sacstart(p+1), vel(EyeData(tr).sacstart(p+1)), 'g*', 'MarkerSize', 10);
%         plot(EyeData(tr).sacend(p), vel(EyeData(tr).sacend(p)), 'c*', 'MarkerSize', 10);
%         pause
%     end
    
    % Compute and bin saccade direction into 8 directions
    % Direction is measured w.r.t horizontal axis
    % Directions are [0 1 2 3 4 5 6 7]*pi/4
    for p=1:length(EyeData(tr).sacend)
    if(EyeData(tr).sacstart(p) > 0 && EyeData(tr).sacend(p) > 0)
      xdisp = EyeData(tr).pixeyeh(EyeData(tr).sacend(p)) - EyeData(tr).pixeyeh(EyeData(tr).sacstart(p));
      ydisp = EyeData(tr).pixeyev(EyeData(tr).sacend(p)) - EyeData(tr).pixeyev(EyeData(tr).sacstart(p));
      theta = atan2(ydisp, xdisp);
      if(theta < 0) theta = 2*pi + theta; end
      theta = round(theta/pi*4);
      if(theta == 8) theta = 0; end
      EyeData(tr).sacdir(p) = theta;
    end
    end
    
    
    
    
    %%%pause
    %%%close(2);
    %%%close(1);
    
end
end
        
        