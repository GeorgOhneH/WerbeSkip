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
python manage.py collectstatic --noinput
echo 'Done...'

echo 'Run migrations'
python manage.py migrate
echo 'Done...'

echo 'Start Redis'
redis-server &
echo 'Done...'

echo 'Start Django'
export PORT=8000
echo 'Server runnning on port ' $PORT
python manage.py runserver &

echo 'Start update_handler.py'
python update_handler.py &

read var1

kill $(ps aux | grep '[p]ython manage.py' | awk '{print $2}')
kill $(ps aux | grep '[r]edis' | awk '{print $2}')
kill $(ps aux | grep '[p]ython update_handler.py' | awk '{print $2}')
