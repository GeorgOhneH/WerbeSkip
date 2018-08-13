# WerbeSkip
A Webserver to detect ads on TV

Build with Machine Learning, Django and Vue

## Installation
Note: Cloning this repository may take a while, because it's very big

[Redis](https://redis.io/download) must be installed on your system
``` bash
# install dependencies
npm install

# serve with hot reload at localhost:8080
# note that the websocket server won't function
npm run dev

# build for production with minification
npm run build

# build for production and view the bundle analyzer report
npm run build --report

# deploy
# Note to stop the bash just press enter, else the processes it starts aren't going close
.deploy.sh
```

## File Structure
There are 2 parts to the app. The first part is the neural network and
the algorithmen. The second part is the webserver.
### Neural Network
The main part is the deepnet directory. It holds a self written deep
neural network library.

In the helperfunction directory are the main function, which are used
to determine if an image is an ad and other helpful functions.
Note: some function need images to function, which aren't on github
[more info](helperfunctions/prosieben)

The numpywrapper module is just so the neural network library can
switch between numpy and cupy

### Webserver
The main parts are: app, src, vuedj
##### vuedj
The Django settings
##### app
Backend and Websocket
##### src
Frontend implemented with VueJS

### Others
thoughts is the directory with the Protokoll

update_handler.py is the intersection between the server and the algorithmen
