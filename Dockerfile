FROM node:alpine
FROM python:3
ENV PYTHONUNBUFFERED 1
RUN apk update && apk upgrade

RUN mkdir /code
WORKDIR /code
ADD . /code

# Install python, pip and python packages
RUN pip install -r requirements.txt

# Run the following commands for deployment
RUN npm set progress=false && npm install -s --no-progress
RUN npm run build
RUN python format_index_html.py
RUN python manage.py collectstatic --noinput

# EXPOSE port to be used
EXPOSE 8000
