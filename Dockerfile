# docker system prune
FROM python:3.6
ENV PYTHONUNBUFFERED 1
ENV DJANGO_DEBUG false
ENV PORT 80

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

COPY build build
COPY config config
COPY static static
COPY src src

COPY .postcssrc.js .
COPY favicon.ico .
COPY index.html .
COPY .babelrc .

RUN npm run build

COPY vuedj vuedj
COPY app app
COPY deepnet deepnet
COPY helperfunctions helperfunctions
COPY numpywrapper numpywrapper

COPY format_index_html.py .
COPY manage.py .
COPY settings_secret.py .

RUN python format_index_html.py
RUN python manage.py collectstatic --noinput --clear

COPY docker-entrypoint.sh .
COPY update_handler.py .

