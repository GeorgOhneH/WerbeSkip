// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import VueNativeSock from 'vue-native-websocket'


Vue.config.productionTip = false


const ws_scheme = window.location.protocol === "https:" ? "wss" : "ws";
const ws_path = ws_scheme + '://' + window.location.host + "/chat/stream/";
Vue.use(VueNativeSock, ws_path, { format: 'json', reconnection: true })
console.log("connected")

new Vue({
  el: '#app',
  router,
  template: '<App/>',
  components: { App }
})
