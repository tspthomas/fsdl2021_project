#!/bin/bash
for class in 'buildings' 'forest' 'glacier' 'mountain' 'sea' 'street'
do
   if [ "$(ls -A ${FEEDBACK_DATA_DIR}intel_image_scene/$class/)" ]; then
      mv ${FEEDBACK_DATA_DIR}intel_image_scene/$class/* ${RAW_DATA_DIR}intel_image_scene/$class/
   fi
done
