% This program takes as an input a text file from the Python function
% beamformer_input_singleframe() found in input_to_beamformer.py and adds 
% an audio activity for each frame found in that text file the audio
% activity of each object using beamforming. This program computes the 
% audio activities for each video frame independently.


% Setup parameters
setup.ffmpegpath = 'E:/Shutter_Encoder/Library'; % In case you don't have FFMPEG installed in some system path
filename_video_detections_txt_file = 'test_singleframe.txt';
filename_output = 'test_output_singleframe.txt';


% Read text file row by row, save the data into data_array
fileID = fopen(filename_video_detections_txt_file,'r');
tline = fgetl(fileID);
data_array = {};
i = 1;
while ischar(tline)
    data_array{i} = tline;
    tline = fgetl(fileID);
    i = i + 1;
end
fclose(fileID);

% Find out the frame index of the last frame
last_frame = split(data_array{end}, ';');
last_frame_index = str2num(last_frame{2});

% Find out the name of the video file
videoname = str2num(last_frame{1});
videoname_sys = ['"' videoname '"'];

% Extract WAV file from video
audioname = [videoname(1:end-4) '.wav'];
audioname_sys = ['"' audioname '"'];
vid2aud_str = sprintf([setup.ffmpegpath '/ffmpeg -i %s -vn -c:a pcm_s16le -y -filter_complex "channelmap=map=FL-FL|FR-FR|FC-BC|BC-FC:channel_layout=4.0" %s'], videoname_sys, audioname_sys);
system(vid2aud_str)

% Read WAV file
%[y,fs] = audioread('R01_10fps.wav');
[y,fs] = audioread(audioname);

% A threshold to define whether there is audio activity or not
audio_threshold = 100;

% Go through each detected object (by the video detector) in the video. To
% get complete audio frames, we do not iterate through the last video frame.
audio_activities = [];
for k = 1:length(data_array)
    line = data_array{k};
    line_elements = split(line, ';');
    video_filename = line_elements{1};
    frame_index = str2num(line_elements{2});
    class_id = line_elements{3};
    class_name = line_elements{4};
    azimuth = str2num(line_elements{5});
    elevation = str2num(line_elements{6});
    video_fps = str2num(line_elements{7});
    video_width_pixels = str2num(line_elements{8});
    video_height_pixels = str2num(line_elements{9});
    
    if frame_index ~= last_frame_index
        nSamplesPerFrame = fs/video_fps;
        indeces_of_audio_samples = (nSamplesPerFrame*(frame_index-1) + 1):(nSamplesPerFrame*frame_index);
        audio_of_segment = y(indeces_of_audio_samples, :);
        cropac_spec = FOA_cropac(audio_of_segment, fs, [azimuth elevation], video_fps);
        % Perform thresholding to determine whether there is audio activity or not
        if max(nansum(cropac_spec)) > audio_threshold
            audio_activities(k) = 1;
        else
            audio_activities(k) = 0;
        end
    end
end

% Create a new text file which is similar to the original text file but it
% has audio activities appended.
% Read text file row by row, save the data into data_array
fileID_output = fopen(filename_output,'w');
for i = 1:length(audio_activities)
    original_row = data_array{i};
    new_row = [original_row ';' num2str(audio_activities(i)) '\n'];
    fprintf(fileID_output, new_row);
end
fclose(fileID_output);









%%


function cropac_spec = FOA_cropac(foasig, sig_fs, beam_dirs, dir_fs)
% archontis.politis@tuni.fi
%
% phi, theta: azimuth, elevation coordinates in RHS
%
% Assuming ACN/SN3D convention
%   ch1: omni
%   ch2: dipole-y   
%   ch3: dipole-z
%   ch4: dipole-x
%
% foasig: the FOA recording [lSigx4 array]
% sig_fs: audio sample rate
% beam_dirs: the beamforming directions given at dir_fs framerate [K x 2 array]
%            [azi1 elev1; ...; azi_K elev_K] in rads
% dir_fs:    rate of direction params, has to be smaller than sig_fs and
%            divide it exactly mod(sig_fs,dir_fs)==0
% beampattern: 'cardioid' or 'hypercardioid'

nSamplesPerFrame = sig_fs/dir_fs;
nFrames = size(beam_dirs,1);
lSig = size(foasig,1);

% convert spherical coords to cartesian
beam_xyz = unitSph2cart(beam_dirs);
% interpolate beamforming directions at signal samplerate
beam_xyz_intrp = zeros(nFrames*nSamplesPerFrame, 3);
beam_xyz_prev = beam_xyz(1,:);
idx=0;

for nf=1:nFrames
    beam_xyz_next = beam_xyz(nf,:);
    beam_xyz_temp = interpolateDirectionsN(beam_xyz_prev, beam_xyz_next, nSamplesPerFrame);
    beam_xyz_intrp(idx+(1:nSamplesPerFrame),:) = beam_xyz_temp(1:end-1,:);
    
    beam_xyz_prev = beam_xyz_next;
    idx = idx+nSamplesPerFrame;
end

% truncate to signal length if necessary
if size(beam_xyz_intrp,1)>lSig, beam_xyz_intrp = beam_xyz_intrp(1:lSig,:);
elseif size(beam_xyz_intrp,1)<lSig, beam_xyz_intrp(end+1:lSig,:) = ones(lSig-size(beam_xyz_intrp,1),1)*beam_xyz_intrp(end,:);
end

% form beamweights for ACN/SN3D
beam_weights = beam_xyz_intrp(:,[2 3 1]);
% apply beampattern
beam_weights = [zeros(lSig,1) beam_weights];

% beamform
bfsig = sum(beam_weights.*foasig,2);

% compute STFTs of beamformed signal
winsize = 0.04*sig_fs;
cvxm_nFrames = 11;
cvxm_nBins = 1;
[spec, cvxM] = mrSTFT_cvxM(foasig, winsize, cvxm_nFrames, cvxm_nBins, 0);
E = squeeze((cvxM(1,1,:,:)+cvxM(2,2,:,:)+cvxM(3,3,:,:)+cvxM(4,4,:,:))/2);
[spec2, cvxM2] = mrSTFT_cvxM([foasig(:,1) bfsig], winsize, cvxm_nFrames, cvxm_nBins, 0);
C = squeeze(real(cvxM2(2,1,:,:)));
C = C.*(C>0);
cropac_spec = C./E;

end

function [xyz] = unitSph2cart(aziElev)
%UNITSPH2CART Get coordinates of unit vectors from azimuth-elevation
%   Similar to sph2cart, assuming unit vectors and using matrices for the 
%   inputs and outputs, instead of separate vectors of coordinates.

if size(aziElev,2)~=2, aziElev = aziElev.'; end
    
[xyz(:,1), xyz(:,2), xyz(:,3)] = sph2cart(aziElev(:,1), aziElev(:,2), 1);

end

function interp_dirs_xyz = interpolateDirectionsN(dir1_xyz, dir2_xyz, N)
% archontis.politis@tuni.fi
%
% interpolate two direction vectors by performing spherical (rotational) 
% linear interpolation, generating direction vectors in between the two,
% across the great circle passing from them. The angle between the two is
% divided in N intervals, hence N+1 vectors are returned, including the two
% original ones.

angle12 = acos(dir1_xyz * dir2_xyz');
cross12 = cross(dir1_xyz, dir2_xyz);
purpvec12 = cross12/sqrt(sum(cross12.^2));

if angle12>0
    dtheta = angle12/N;
    psi = 0:dtheta:angle12;
    for k = 1:length(psi)
        interp_dirs_xyz(k,:) = dir1_xyz * cos(psi(k)) + cross(purpvec12, dir1_xyz)*sin(psi(k)) + purpvec12*cross(purpvec12, dir1_xyz)'*(1-cos(psi(k)));
    end
else
    interp_dirs_xyz = ones(N+1,1)*dir1_xyz;
end

end


function [spectrum, cvxM] = mrSTFT_cvxM(insig, winsize, cvxm_nFrames, cvxm_nBins, VERBOSE)

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
