function [onset, offset, stimID, condID] = getTrials4(nev)

% Look for the sequence 200 (trial onset) 10 (stim onset) 40 (stim offset)
% If the sequence does not exist for some trials, discard them
trialonset = find(nev(:,1) == 0 & nev(:,2) == 1);

for j=1:length(trialonset)-1
  onset_entry = find(nev(trialonset(j):trialonset(j+1), 2) == 10);
  offset_entry = find(nev(trialonset(j):trialonset(j+1), 2) == 40);
  if(~isempty(onset_entry) && ~isempty(offset_entry))
    onset(j) = round(1000*nev(trialonset(j)+onset_entry(1), 3));
    offset(j) = round(1000*nev(trialonset(j)+offset_entry(1), 3));
  else
    fprintf('Trial skipped: %d\n', j);
  end
end

onset_entry = find(nev(trialonset(j+1):end, 2) == 10);
offset_entry = find(nev(trialonset(j+1):end, 2) == 40);
if(~isempty(onset_entry) && ~isempty(offset_entry))
  onset(j+1) = round(1000*nev(trialonset(j+1)+onset_entry(1), 3));
  offset(j+1) = round(1000*nev(trialonset(j+1)+offset_entry(1), 3));
else
  fprintf('Trial skipped: %d\n', j);
end

trialonset = find(nev(:,1) == 0 & nev(:,2) == 1);
trialoffset = find(nev(:,1) == 0 & nev(:,2) == 255);
code_idx = find(nev(:,1) == 0 & nev(:,2) > 255);

for tr=1:length(trialoffset)
    this_trial_code_idx = code_idx(code_idx >= trialonset(tr)+1 & code_idx < trialoffset(tr));
    trialmeta(tr).data = char(nev(this_trial_code_idx,2)-256)';
    fprintf('%s\n', trialmeta(tr).data);
    
    t = regexp(trialmeta(tr).data, 'suffix=(\w*)', 'tokens');
    tt = t{1,1};
    stimID(tr) = str2num(tt{1,1}(1:end));
    
    t = regexp(trialmeta(tr).data, 'movieprefix=(\w*)/(\w*)/(\w*)', 'tokens');
    tt = regexp(t{1}(3), '_', 'split');
    tmp = tt{1}(1);
    condID{tr} = tmp{1};
end

% Make sure they are all the same length
if(numel(stimID) < numel(onset))
    onset(numel(stimID)+1:end) = [];
    offset(numel(stimID)+1:end) = [];
end