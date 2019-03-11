#!/bin/bash

python manage.py makemigrations
python manage.py migrate --noinput
daphne -b 0.0.0.0 -p 80 vuedj.asgi:application
