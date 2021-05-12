#!/bin/bash
for class in 'buildings' 'forest' 'glacier' 'mountain' 'sea' 'street'
do
   if [ "$(ls -A ${FEEDBACK_DATA_DIR}intel_scene_images/$class/)" ]; then
      cp ${FEEDBACK_DATA_DIR}intel_scene_images/$class/* ${RAW_DATA_DIR}intel_scene_images/seg_train_1000/$class/
   fi
done
