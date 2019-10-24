function [stimonset, stimoffset] = clean_onset_offset_times(stimonset, stimoffset, lastspike)

% Clean it
toDel = [];
new_stimoffset = [];
for n=1:length(stimonset)
    nearest_offset = stimoffset(stimoffset > stimonset(n));

    if(isempty(nearest_offset))
        toDel = [toDel; n];
        continue;
    end

    if(nearest_offset(1) > lastspike)
        toDel = [toDel; n];
        continue;
    end
    
    if(numel(nearest_offset > 2))
        nearest_offset(2:end) = [];
    end

    new_stimoffset = [new_stimoffset; nearest_offset];
end

stimonset(toDel) = [];
stimoffset = new_stimoffset;
