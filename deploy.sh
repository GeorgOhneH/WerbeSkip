#!/usr/bin/env bash
echo 'Run npm build'
npm run build
echo 'Done...'

echo 'Format index.html as Jinja template'
python format_index_html.py
echo 'Done...'

echo 'Install python modules'
pip install -r requirements.txt
echo 'Done...'

echo 'Collect static'
python manage.py collectstatic --noinput --clear
echo 'Done...'
python manage.py makemigrations
echo 'Run migrations'
python manage.py migrate
echo 'Done...'

echo 'Start Redis'
redis-server &
echo 'Done...'
echo 'Start Django'
export PORT=8000
echo 'Server runnning on port ' $PORT
daphne -b 0.0.0.0 -p 8000 vuedj.asgi:application &

echo 'Start update_handler.py'
python update_handler.py &

read
