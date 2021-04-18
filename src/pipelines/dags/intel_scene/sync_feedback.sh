#!/bin/bash
for class in 'buildings' 'forest' 'glacier''mountain' 'sea' 'street'
do
   mv $FEEDBACK_DATA_DIR/intel_image_scene/seg_train/seg_train/$class/* $RAW_DATA_DIR/intel_image_scene/seg_train/seg_train/$class/
done
