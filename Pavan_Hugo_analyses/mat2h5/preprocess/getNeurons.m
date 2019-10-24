function [InfoNeuron2, SpikeTimes2]=getNeurons(nev)
idx_valid=find(~ismember(nev(:,2),[0,255]));
UNITS=setdiff(unique(nev(:,1)),0);
Nneuron=0;
for u=1:length(UNITS)
    unit=UNITS(u);
    idx_unit=intersect(find(nev(:,1)==unit),idx_valid);
    NEURONS=unique(nev(idx_unit,2));
    for n=1:length(NEURONS)
        neuron=NEURONS(n);
        idx_neu=intersect(find(nev(:,2)==neuron),idx_unit);
        
        % Reject neurons that spike at < 1 Hz
        if length(nev(idx_neu,3))/length(nev)*1000 > 1
            Nneuron=Nneuron+1;
            SpikeTimes{Nneuron}=nev(idx_neu,3);
            InfoNeuron{Nneuron}=[unit neuron];
        end
    end
end


InfoNeuron2=[];
c=0;

for i=1:length(InfoNeuron)
  neuronsperunit(i) = InfoNeuron{i}(2);
end
MAXNEURONS = max(neuronsperunit);

for unitneuron=1:MAXNEURONS
for neuron=1:length(InfoNeuron)
    if InfoNeuron{neuron}(2)==unitneuron
        c=c+1;
        InfoNeuron2=[InfoNeuron2;InfoNeuron{neuron}];
        SpikeTimes2{c}=SpikeTimes{neuron};
    end
end
end
