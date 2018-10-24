# docker system prune
FROM python:3.6
ENV PYTHONUNBUFFERED 1
ENV DJANGO_DEBUG false

ADD ffmpeg /usr/local/bin

RUN mkdir /code
WORKDIR /code

#install npm
RUN apt-get update \
    && apt-get upgrade -y \
    && curl -sL https://deb.nodesource.com/setup_8.x | bash - \
    && apt-get install -y nodejs


COPY package.json package.json
COPY package-lock.json package-lock.json

RUN npm install

ADD requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN npm run build

RUN python format_index_html.py
RUN python manage.py collectstatic --noinput
