function [onset, offset, stimID, condID] = getTrials2(nev)

% Identify all the trial onsets
trialonset = find(nev(:,1) == 0 & nev(:,2) == 1);

%trialoffset = find(nev(:,2) == 201);

% Look for the sequence 200 (trial onset) 10 (stim onset) 11 (stim offset)
% If the sequence does not exist for some trials, discard them

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

stimID = mod(nev(trialonset+1,2)-2^15, 300);
stimID((stimID == 0)) = 300;
condID = floor((nev(trialonset+1,2)-2^15)/300);
