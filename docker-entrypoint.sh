#!/bin/bash

python manage.py migrate --no-input

daphne -b 0.0.0.0 -p 8000 vuedj.asgi:application
