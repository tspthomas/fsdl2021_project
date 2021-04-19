#!/bin/bash
for class in 'buildings' 'forest' 'glacier' 'mountain' 'sea' 'street'
do
   if [ "$(ls -A $FEEDBACK_DATA_DIR/$class/)" ]; then
      mv $FEEDBACK_DATA_DIR/$class/* $RAW_DATA_DIR/intel_image_scene/seg_train/seg_train/$class/
   fi
done
