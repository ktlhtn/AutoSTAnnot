% Convert audio powermap from MATLAB into an MP4 video.

load('processed_0_map_scaled_R0010861.mat')

% Load video
v = map_scaled;

[~,~,frames] = size(map_scaled);

writer = VideoWriter('test.mp4', 'MPEG-4');
writer.FrameRate = 10;
open(writer)

for i = 1:frames
    writeVideo(writer, map_scaled(:,:,i)/max(max(max(map_scaled))));
end

close(writer);