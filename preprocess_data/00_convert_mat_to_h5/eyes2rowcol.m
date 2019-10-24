function [row, col] = eyes2rowcol(eyeh, eyev, screenDistance, pixPerCM, HEIGHT, WIDTH)

% Degrees to pixels
%screenDistance = 36;
%pixPerCM = 27.03;
pixeyev = HEIGHT/2 - pixPerCM*screenDistance*tan(eyev*pi/180);
pixeyeh = WIDTH/2 + pixPerCM*screenDistance*tan(eyeh*pi/180);

row = round(pixeyev);
col = round(pixeyeh);