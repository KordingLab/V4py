function [betaX, betaY] = test_calibration_data(filepath, filename)

HEIGHT = 1000;
WIDTH = 1000;

%% 
% Load the nev file
load(sprintf('%s/nev/%s_nev.mat', filepath, filename));

% Load the eye file
load(sprintf('%s/eyes/%s_eyes.mat', filepath, filename));

eyeh = eyex(1:3:end);
eyev = eyey(1:3:end);

%% Get voltage readings and pixel readings for saccade landings after each trial
codes = nev(nev(:,1)==0,2:3);
trialonset = find(codes(:,1) == 1);
trialoffset = find(codes(:,1) == 255);

pixX = []; pixY = []; voltX = []; voltY = [];

for tr=1:numel(trialonset)
    trialstart = trialonset(tr);
    tmp = find(trialoffset > trialonset(tr));
    trialend = trialoffset(tmp(1));
    trialcodes = codes(trialstart:trialend, :);
    
    % Discard the trial if a saccade is not made
    if(sum(trialcodes(:,1) == 140) ~= 2)
        continue;
    end
    
    % Get the saccade angle
    fx_times = round(1e3*trialcodes(trialcodes(:,1) == 140, 2));
    trialmsg = char(trialcodes(trialcodes(:,1) > 255 & trialcodes(:,1) < 512, 1)-256)';
    variables = regexp(trialmsg, ';', 'split');
    theta = [];
    for v=1:numel(variables)
        if(~isempty(regexp(variables{v}, '|saccadeDir', 'once')))
            tmp = regexp(variables{v}, '=', 'split');
            theta = [theta; pi/180 * str2double(tmp{2})];
        end
    end
    theta = theta(end);        
    
    % Center fixations
    voltX = [voltX; eyeh(fx_times(1))];
    voltY = [voltY; eyev(fx_times(1))];
    pixX = [pixX; WIDTH/2];
    pixY = [pixY; HEIGHT/2];
    
    % Outer fixations
    voltX = [voltX; eyeh(fx_times(2))];
    voltY = [voltY; eyev(fx_times(2))];
    pixX = [pixX; WIDTH/2 + 150 * cos(theta)];
    pixY = [pixY; HEIGHT/2 - 150 * sin(theta)];
    
    
end

%% Regress voltages on to pixels
betaX = glmfit(voltX, pixX);
betaY = glmfit(voltY, pixY);

