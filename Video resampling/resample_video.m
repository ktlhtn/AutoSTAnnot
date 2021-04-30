% Resample videos into a different framerate using ffmpeg

% Initial parameters
video_read_dir = 'C:\TTY\SP and ML\SP innovation project\own_recordings_ricoh_theta\recordings240321\converted_er_spatial_audio';
video_name = 'R0010833_er.mov';
video_write_dir = 'C:\TTY\SP and ML\SP innovation project\own_recordings_ricoh_theta\recordings240321\converted_er_spatial_audio';
video_write_name = 'R0010833_er_10fps.mp4';
ffmpeg_dir = 'E:/Shutter_Encoder/Library';
target_fps = 10;

% Do the actual conversion
video_name_with_dir = [video_read_dir filesep video_name];
video_name_sys = ['"' video_name_with_dir '"'];
output_video_with_dir = [video_write_dir filesep video_write_name];
video_write_name_sys = ['"' output_video_with_dir '"'];
command = sprintf([ffmpeg_dir '/ffmpeg -i %s -filter:v fps=%s %s'], video_name_sys, string(target_fps), video_write_name_sys);
system(command)