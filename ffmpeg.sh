ffmpeg -r 60 -f image2 -s 1920x1080 -i frame%04d_annotated.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p output.mp4

cat *_annotated.jpg | ffmpeg -f image2pipe -i - output.mp4

ffmpeg -framerate 30 -pattern_type glob -i '*._annotatedpng' \
  -c:v libx264 -pix_fmt yuv420p out.mp4