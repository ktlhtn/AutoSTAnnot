% Convert an input video with B-format audio into an audio powermap using
% the MUSIC algorithm.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Written by Archontis Politis and Sharath Adavanne, 3/2017
%   TUT, Finland
%
%   Modified for Ricoh Theta V by Archontis Politis, 3/2020
%   TAU, Finland
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

% REQUIREMENTS
% https://github.com/polarch/Spherical-Harmonic-Transform
% https://github.com/polarch/Vector-Base-Amplitude-Panning
% https://github.com/polarch/Spherical-Array-Processing
% Put these in some folder that Matlab searches at (e.g. the datapath folder)

% Ricoh specific options
UPSIDE_DOWN = 0;

% setup structure with video, datapaths etc...
setup.datapath = './resources/'; % path of grid data
setup.vidpath = 'C:\TTY\SP and ML\SP innovation project\own_recordings_ricoh_theta\recordings_13042021\converted_videos_er_spatial_audio';
setup.ffmpegpath = 'E:/Shutter_Encoder/Library'; % In case you don't have FFMPEG installed in some system path
% that Matlab looks at automatically
% (e.g. /usr/local/bin), because for example you
% don't have privileges to do so, use this to
% specify the local path where FFMPEG is. You can
% check if MAtlab can find FFMPEG by running
% >> !ffmpeg
% If it is found, then leave this path empty.

setup.WRITE_MAP   = false;  % Write the acoustic map as a video
setup.WRITE_VIDEO_WITH_MAP = false;  % Write the combined video with map overlay
setup.PREVIEW_MAP = true; % Preview the power map in Matlab
setup.VERBOSE     = true;  % Spit out some info

setup.projectname = 'R0010861'; % That should be the name of the specific recording (e.g. Ricoh_test_er.mov)

setup.method = 'MUSIC'; % Method for the spatial powermap - options are 'PWD','MUSIC','AIV'
% setup.videotext = ''; % Text to overlay on the video, leave empty or do
% not define field if you don't want this. IT NEEDS
% FFMPEG BUILT WITH FREETYPE SUPPORT!

[audio, afs] = prepare_audio_video(setup);
[~, ~, map, map_scaled] = generate_powermap(setup, audio, afs);


% These are the audio powermaps (scaled and non-scaled), which can be used
% as an input to the fuctions in the directory detections_to_powermap.
save('map.mat')
save('map_scaled.mat')


%%
function [camsig_res, fsa] = prepare_audio_video(setup)

% paths
videoname = [setup.vidpath filesep setup.projectname '_er.mov'];
videoname_sys = ['"' videoname '"'];

% get camera audio and resample at 24kHz
if (setup.VERBOSE) disp('Load audio from video and downsample.'), end
audioname = [videoname(1:end-4) '.wav'];
audioname_sys = ['"' audioname '"'];
vid2aud_str = sprintf([setup.ffmpegpath '/ffmpeg -i %s -vn -c:a pcm_s16le -y -filter_complex "channelmap=map=FL-FL|FR-FR|FC-BC|BC-FC:channel_layout=4.0" %s'], videoname_sys, audioname_sys);
system(vid2aud_str)
[camsig, fs_cam] = audioread(audioname);

fsa = 24000;
camsig_res = resample(camsig, fsa, fs_cam); % downsample

if setup.WRITE_VIDEO_WITH_MAP
    if setup.VERBOSE, disp('Convert video to grayscale and downsample.'), end
    videoname_bw = [videoname(1:end-4) '_bw.mp4'];
    videoname_bw_sys = ['"' videoname_bw '"'];
    vid2vid_str = sprintf([setup.ffmpegpath '/ffmpeg -i %s -an -c:v h264 -y -vf scale=960:-1,fps=10,hue=s=0 %s'], videoname_sys, videoname_bw_sys);
    system(vid2vid_str)
end

% truncate the original multichannel file to the proper length
%     start_sample = delay_samples_org_fs;
%     end_sample = delay_samples_org_fs+round(lVideo*fs_spat/fs_cam);
%     if (setup.VERBOSE), disp('Trim original multichannel recording.'), end
%     audioname_out = [audioname_in(1:end-4) '_aligned.wav'];
%     audioname_out_sys =['"' audioname_out '"'];
%     trimAud_str = sprintf([setup.ffmpegpath 'ffmpeg -i %s -af atrim=start_sample=%d:end_sample=%d -acodec pcm_s24le -y %s'], audioname_in_sys, start_sample, end_sample, audioname_out_sys);
%     system(trimAud_str)

end

%%
function [pmap_broad, doa_hist, map, map_scaled] = generate_powermap(setup, audio, afs)

vidpath = setup.vidpath; % path of Ricoh Theta video
projectname = setup.projectname;
videoname = [setup.vidpath filesep projectname '_er.mov'];
videoname_sys = ['"' videoname '"'];
videoname_bw = [videoname(1:end-4) '_bw.mp4'];
videoname_bw_sys = ['"' videoname_bw '"'];

% fix Ricoh channel order
audio = audio(:,[1 2 4 3]);
% convert from SN3D to N3D
audio(:,2:4) = audio(:,2:4)*sqrt(3);

winlength   = 0.04; % sec
winsize     = winlength*afs;

[spectrum, cvxM] = stft_cvxM(audio, winsize, 11, 5);
binfreqs = (0:winsize/2)'*afs/winsize;
frametimes = (0:size(spectrum,2))*winsize/2/afs;

% keep 10 SCMs per second for analysis, to match the video framerate
nFramesPerSecond = afs/(winsize/2);
cvxM_10 = cvxM(:,:,:,1:nFramesPerSecond/10:end);
% permute cvxM to [nCH x nCH x nFrames x nBins]
cvxM_10 = permute(cvxM_10, [1 2 4 3]);
% keep only SCMs of a preferred frequency range
cvxmf_lims = [200 8000];
[~,idxf_lims(1)] = min(abs(cvxmf_lims(1)-binfreqs));
[~,idxf_lims(2)] = min(abs(cvxmf_lims(2)-binfreqs));
cvxM_10 = cvxM_10(:,:,:,idxf_lims(1):idxf_lims(2));

% compute powermap
[doagrid_xyz, doagrid_dirs] = getFliegeNodes(30);
nGrid = size(doagrid_dirs,1);
grid_steervecs = sqrt(4*pi)*getRSH(1, doagrid_dirs*180/pi);
histSmoothCoeff = 0.5;
% histogram only single-source-DoAs
SINGLE_SRC_ONLY = 1;
% Compute powermap
nSH = size(cvxM_10,1);
nFrames_10 = size(cvxM_10,3);
nBins_10 = size(cvxM_10,4);
maxnSrc = floor(nSH/2);

pmap_broad = zeros(nGrid,nFrames_10);
doa_hist = zeros(nGrid,nFrames_10);
doa_hist_prev = zeros(nGrid,1);
for nf=1:nFrames_10
    doa_hist_nf = zeros(nGrid,1);
    for nb=1:nBins_10
        cvxM_nf_nb = cvxM_10(:,:,nf,nb);
        
        [V,S] = sorted_eig(cvxM_nf_nb);
        lambda = diag(real(S));
        nSrc = sorte(lambda);
        if nSrc>maxnSrc, nSrc = maxnSrc; end
        Vn = V(:,nSrc+1:end);
        pmap_nb = sphMUSIC(grid_steervecs, Vn);
        pmap_broad(:,nf) = pmap_broad(:,nf) + pmap_nb/max(pmap_nb);
        
        if SINGLE_SRC_ONLY
            if nSrc==1
                doa_idx = peakFind2d(pmap_nb, doagrid_xyz, 1);
            else
                doa_idx = 0;
            end
        else
            if nSrc
                doa_idx = peakFind2d(pmap_nb, doagrid_xyz, nSrc);
            else
                doa_idx = 0;
            end
        end
        if doa_idx
            doa_hist_nf(doa_idx) = doa_hist_nf(doa_idx)+1;
        end
    end
    doa_hist(:,nf) = histSmoothCoeff*doa_hist_prev + (1-histSmoothCoeff)*doa_hist_nf;
    doa_hist_prev = doa_hist(:,nf);
end

%%% Interpolate audio map from uniform grid to rectangular image (1pixel/deg)
% Triangular interpolation

%%% SLOW!!! Load saved interpolation table if it has been computed
%%% previosuly
if ~exist('Tri2EquirectInterpTable_N900_E1A1.mat','file')
    if (setup.VERBOSE) disp('Calculating triangular-to-rectangular interpolation map, from beamforming grid to pixels.'), end
    azi_res = 1;
    elev_res = 1;
    N_azi = round(360/azi_res) + 1;
    gtable3D = getGainTable(doagrid_dirs*180/pi, [azi_res elev_res]);
    gtable_norm = sum(gtable3D,2);
    interpTable = gtable3D./repmat(gtable_norm, [1 nGrid]);
    
    save([datapath 'Tri2EquirectInterpTable_N900_E1A1.mat'],'interpTable');
else
    if (setup.VERBOSE) disp('Loading triangular-to-rectangular interpolation map, from beamforming grid to pixels.'), end
    load('Tri2EquirectInterpTable_N900_E1A1')
end

mapSmoothCoeff = 0.8;
nAFrames = nFrames_10; % number of powermap video frames

if (setup.VERBOSE) disp('Compute and smooth interpolated acoustic map.'), end
map = interpTable*pmap_broad;
map = reshape(map, [361 181 nAFrames]);
map = permute(map, [2 1 3]);
map = map(end:-1:1,:,:); % flip up-down
map = map(:,end:-1:1,:); % flip left-right


mapmin = min(min(min(map)));
map_scaled = map-mapmin;
mapmax = max(max(max(map_scaled)));
map_scaled = map_scaled/mapmax;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Change the scale of the powermap into 4k
%new_height = 1920;
%new_width = 3840;

%[~,~,vid_len] = size(map);
%map_temp = map;
%map = zeros(new_height,new_width,vid_len);

%for frame=1:vid_len
%    map(:,:,frame) = imresize(map_temp(:,:,frame), [new_height new_width]);
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


map_prev = 0;
for naf=1:nAFrames
    map_scaled(:,:,naf) = mapSmoothCoeff*map_prev + (1-mapSmoothCoeff)*map_scaled(:,:,naf);
    map_prev = map_scaled(:,:,naf);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A threshold for detected sounds
%sound_threshold = 10*min(min(min(map_scaled))); % Can be anything
%map_scaled(map_scaled < sound_threshold) = 0;
%map_scaled(map_scaled >= sound_threshold) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if setup.PREVIEW_MAP
    frametimes_10 = (0:nAFrames-1)*0.1;
    if (setup.VERBOSE) disp('Preview acoustic map.'), end
    figure
    for naf=1:nAFrames
        imagesc(map_scaled(:,:,naf)), colorbar, axis equal
        title(sprintf('frame: %4d / sec: %2.2f', naf, frametimes_10(naf)))
        drawnow
    end
end

%%% WRITE POWERMAP
if setup.WRITE_MAP
    cmap = colormap('jet');
    cmap = interp1(1:size(cmap,1), cmap, linspace(1,size(cmap,1),256));
    close
    
    if (setup.VERBOSE) disp('Combine video with acoustic map.'), end
    height = 480;
    width = 2*height;
    writer = VideoWriter([vidpath filesep 'temp_map.avi'], 'Motion JPEG AVI');
    writer.FrameRate = 10;
    open(writer)
    normalize_caxis = 1;
    counter = 1;
    progress_prev = 0;
    for naf=1:nAFrames
        audframe = map_scaled(:,:,counter);
        if normalize_caxis
            audmin = min(min(audframe));
            audmax = max(max(audframe));
            audframe = (audframe - audmin)/(audmax - audmin);
        end
        audframe = imresize(audframe, [height width]);
        audframe = im2uint8(ind2rgb(round(audframe*255)+1, cmap));
        writeVideo(writer, audframe);
        
        if (setup.VERBOSE)
            progress = round(100 * counter/nAFrames);
            if progress ~= progress_prev
                disp(['         Writing video: Progress ' num2str(progress) '%'])
                progress_prev = progress;
            end
        end
        counter = counter +1;
    end
    close(writer);
end

%%% Combine video and audio map
if setup.WRITE_VIDEO_WITH_MAP
    
    % Load video
    if (setup.VERBOSE) disp('Loading Ricoh Theta video.'), end
    v = VideoReader(videoname_bw);
    
    %frame counter
    if (setup.VERBOSE) disp('       Counting number of video frames.'), end
    fcounter = 0;
    while hasFrame(v)
        readFrame(v);
        fcounter = fcounter+1;
    end
    v.CurrentTime = 0;
    progress_prev = 0;    
    
    vfs = v.FrameRate;  % video framerate
    nVFrames = fcounter;
    vt = linspace(0, v.Duration, nVFrames);
    
    % make video and audio map same length if not already
    if nAFrames<nVFrames
        map_scaled(:,:,end+1:nVFrames) = repmat(map_scaled(:,:,end), [1 1 nVFrames-nAFrames]);
    end
    
    cmap = colormap('jet');
    cmap = interp1(1:size(cmap,1), cmap, linspace(1,size(cmap,1),256));
    close
    
    if (setup.VERBOSE) disp('Combine video with acoustic map.'), end
    height = 480;
    width = 2*height;
    writer = VideoWriter([vidpath filesep 'temp.avi'], 'Motion JPEG AVI');
    writer.FrameRate = vfs;
    open(writer)
    normalize_caxis = 1;
    counter = 1;
    while hasFrame(v)
        vidframe = rgb2gray(readFrame(v));
        audframe = map_scaled(:,:,counter);
        if normalize_caxis
            audmin = min(min(audframe));
            audmax = max(max(audframe));
            audframe = (audframe - audmin)/(audmax - audmin);
        end
        %audframe = imresize(audframe, [size(vidframe,1), size(vidframe,2)]);
        audframe = imresize(audframe, [height width]);
        vidframe = imresize(vidframe, [height width]);
        audframe = im2uint8(ind2rgb(round(audframe*255)+1, cmap));
        avframe = imlincomb(0.7, repmat(vidframe, [1 1 3]), 0.4, audframe);
        writeVideo(writer, avframe);
        
        if (setup.VERBOSE)
            progress = round(100 * counter/nVFrames);
            if progress ~= progress_prev
                disp(['         Writing video: Progress ' num2str(progress) '%'])
                progress_prev = progress;
            end
        end
        counter = counter +1;
    end
    close(writer);
end

%% Convert AVI to compressed MP4 using ffmpeg and insert mono audio stream from EM32
if setup.WRITE_MAP
    if (setup.VERBOSE) disp('Compress video (H264/AAC/MP4) and inject mono audio.'), end
    videoname_out = [vidpath filesep projectname '_map.mp4'];         % Ricoh theta equirectangular video recording
    tempname = [vidpath filesep 'temp_map.avi'];
    % apply text overlay if specified (omit the field, or leave empty if you
    % don't want it) WARNING!!! It needs freetype support, and a build of FFMPEG
    % with it
    if isfield(setup,'videotext') && ~isempty(setup.videotext)
        ffmpeg_str = sprintf([setup.ffmpegpath '/ffmpeg ' ...
            '-i %s -i %s -y ' ...
            '-vf "drawbox=y=ih/2:color=black@0.4:width=iw:height=48:t=max, drawtext=fontfile=OpenSans-Regular.ttf:text=''%s'':fontcolor=white:fontsize=24:x=(w-tw)/2:y=(h/2)+th" ' ...
            '-c:v libx264 -crf 20 -preset fast ' ...
            '-c:a aac -map 0:v:0 -map 1:a:0 ' ...
            '%s'], ...
            tempname, videoname_sys, setup.videotext, videoname_out);
    else
        ffmpeg_str = sprintf([setup.ffmpegpath '/ffmpeg ' ...
            '-i %s -i %s -y ' ...
            '-c:v libx264 -crf 20 -preset fast ' ...
            '-c:a aac -map 0:v:0 -map 1:a:0 ' ...
            '%s'], ...
            ['"' tempname '"'], videoname_sys, ['"' videoname_out '"']);
    end
    system(ffmpeg_str)
    delete(tempname);
end


end

function P_music = sphMUSIC(A_grid, Vn)

VnA = Vn'*A_grid;
P_music = 1./sum(conj(VnA).*VnA);
P_music = P_music.';

end

function est_idx = peakFind2d(pmap, grid_xyz, nPeaks)

kappa  = 50;
P_minus_peak = pmap;
est_idx = zeros(nPeaks, 1);
for k = 1:nPeaks
    [~, peak_idx] = max(P_minus_peak);
    est_idx(k) = peak_idx;
    if (k~=nPeaks)
        VM_mean = grid_xyz(peak_idx,:); % orientation of VM distribution
        VM_mask = kappa/(2*pi*exp(kappa)-exp(-kappa)) * exp(kappa*grid_xyz*VM_mean'); % VM distribution
        VM_mask = 1./(0.00001+VM_mask); % inverse VM distribution
        P_minus_peak = P_minus_peak.*VM_mask;
    end
end
end

function [V,D] = sorted_eig(COV)

[V,D] = eig(COV);
d = real(diag(D));
[d_sorted, idx] = sort(d,'descend');
D = diag(d_sorted);
V = V(:,idx);

end

function [K, obfunc] = sorte(lambda)

if all(lambda==0)
    K = 0;
else
    N = length(lambda);
    Dlambda = lambda(1:end-1) - lambda(2:end);
    for k=1:N-1
        meanDlambda = 1/(N-k)*sum(Dlambda(k:N-1));
        sigma2(k) = 1/(N-k)*sum( (Dlambda(k:N-1) - meanDlambda).^2 );
    end
    for k=1:N-2
        if sigma2(k)>0, obfunc(k) = sigma2(k+1)/sigma2(k);
        elseif sigma2(k) == 0, obfunc(k) = Inf;
        end
    end
    %    [~,K] = min(obfunc(1:end-1));
    [~,K] = min(obfunc);
end
end

function [spectrum, cvxM] = stft_cvxM(insig, winsize, cvxm_nFrames, cvxm_nBins, VERBOSE)
% written by Archontis Politis, archontis.politis@aalto.fi

if ~exist('VERBOSE','var') || isempty(VERBOSE)
    VERBOSE = 1;
end

lSig = size(insig,1);
nCHin = size(insig,2);

% time-frequency processing
fftsize = winsize;
hopsize = winsize/2;
nBins = winsize/2 + 1;
nWindows = ceil(lSig/winsize);
nFrames = 2*nWindows;

% zero pad the signal's start and end for STFT
insig_pad = [zeros(winsize/2, nCHin); insig; zeros(nWindows*winsize-lSig, nCHin)];
clear insig

spectrum = zeros(nBins, nFrames, nCHin);
if nargout>1
    if mod(cvxm_nFrames,2)==0, cvxm_nFrames = cvxm_nFrames+1; end
    if mod(cvxm_nBins,2)==0, cvxm_nBins = cvxm_nBins+1; end
    cvxm_frameBuffer = zeros(nBins, nCHin, cvxm_nFrames);
    cvxM = zeros(nCHin, nCHin, nBins, nFrames);
end

% transform window (hanning)
x = 0:(winsize-1);
win = sin(x.*(pi/winsize))'.^2;

% processing loop
idx = 1;
nf = 1;

while nf <= nFrames
    if VERBOSE
        if mod(nf,10)==0, fprintf('%d/%d\n',nf,nFrames); end
    end
    
    % Window input and transform to frequency domain
    insig_win = win*ones(1,nCHin) .* insig_pad(idx+(0:winsize-1),:);
    inspec = fft(insig_win, fftsize);
    inspec = inspec(1:nBins,:); % keep up to nyquist
    spectrum(:,nf,:) = inspec;
    
    if nargout>1
        % collect new frame for averaging
        cvxm_frameBuffer(:,:,2:end) = cvxm_frameBuffer(:,:,1:end-1);
        cvxm_frameBuffer(:,:,1) = inspec;
        cvxm_nFramesHalf = floor(cvxm_nFrames/2);
        cvxm_nBinsHalf = floor(cvxm_nBins/2);
        
        if nf>cvxm_nFramesHalf
            
            for nb = 1:nBins
                Cxx = zeros(nCHin,nCHin);
                for jj = nb+(-cvxm_nBinsHalf:cvxm_nBinsHalf)
                    if jj>0 && jj<nBins+1
                        in = squeeze(cvxm_frameBuffer(nb,:,:));
                        Cxx = Cxx + in*in'/cvxm_nFrames;
                    end
                end
                Cxx = Cxx/cvxm_nBins;
                cvxM(:,:,nb,nf-cvxm_nFramesHalf) = Cxx;
            end
        end
    end
    % advance sample pointer
    idx = idx + hopsize;
    nf = nf + 1;
    
end
fprintf('%d/%d\n',nf-1,nFrames);

end

function R_N = getRSH(N, dirs_deg)
%GETRSH Get vector of real orthonormal spherical harmonic values up to order N
%
% Inputs:
%   N:      maximum order of harmonics
%   dirs:   [azimuth_1 elevation_1; ...; azimuth_K elevation_K] angles
%           in degs for each evaluation point, where elevation is the
%           polar angle from the horizontal plane
%
% Outpus:
%   R_N:    [(N+1)^2 x K] matrix of SH values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Archontis Politis, 10/10/2013
%   archontis.politis@aalto.fi
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ndirs = size(dirs_deg, 1);
Nharm = (N+1)^2;

% convert to rads
dirs = dirs_deg*pi/180;
% initialize SH matrix
R_N = zeros(Nharm, Ndirs);
% zero order
R_N(1,:) = 1/sqrt(4*pi);
% higher orders
if N>0 
    idx_R = 1;
    for n=1:N
        
        m = (0:n)';
        % vector of unnormalised associated Legendre functions of current order
        Pnm = legendre(n, sin(dirs(:,2)'));
        % cancel the Condon-Shortley phase from the definition of
        % the Legendre functions to result in signless real SH
        uncondon = (-1).^[m(end:-1:2);m] * ones(1,Ndirs);
        Pnm = uncondon .* [Pnm(end:-1:2, :); Pnm];
        
        % normalisations
        norm_real = sqrt( (2*n+1)*factorial(n-m) ./ (4*pi*factorial(n+m)) );
        
        % convert to matrix, for direct matrix multiplication with the rest
        Nnm = norm_real * ones(1,Ndirs);
        Nnm = [Nnm(end:-1:2, :); Nnm];
        
        CosSin = zeros(2*n+1,Ndirs);
        % zero degree
        CosSin(n+1,:) = ones(1,size(dirs,1));
        % positive and negative degrees
        CosSin(m(2:end)+n+1,:) = sqrt(2)*cos(m(2:end)*dirs(:,1)');
        CosSin(-m(end:-1:2)+n+1,:) = sqrt(2)*sin(m(end:-1:2)*dirs(:,1)');
        Rnm = Nnm .* Pnm .* CosSin;
        
        R_N(idx_R + (1:2*n+1), :) = Rnm;
        idx_R = idx_R + 2*n+1;
    end
end

end