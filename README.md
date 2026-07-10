# cs231n-jaspernet
usage: python full_pipeline.py input.csv output.csv

use --photo-root to specify folder root location, in this instance defaults to jasper-wildlife/ct_photos otherwise


input.csv must include Name,CameraPath, see test_input.csv as an example 

output.csv includes:
CameraPath,
Name,
bbox (a list of bounding box coordinates in terms of image ratios),
MetaId,
rough_category (from segmenter),
detection_confidence (from segmenter),
img_w,
img_h,
n_detections,
pred_species (dict from species code to counts),
pred_species_name (dict from species name to counts),
pred_species_conf (list of confidence scores from classifier),
pred_species_topk (list of lists of top 5 species predicted, one list for each detection)

To visualize, use 
python show_detections.py --csv output.csv \
        --camera-path CAMERA_ARRAY_A/camera_A2 --name example.jpeg \
        --images-root jasper-wildlife/ct_photos --out example_annotated.jpg

use --photo-root to specify folder root location, in this instance defaults to jasper-wildlife/ct_photos otherwise