#!/bin/bash

gunicorn app:app -b 0.0.0.0:$FLASK_PORT -w 4 --threads 4