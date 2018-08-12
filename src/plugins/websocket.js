import Vue from 'vue'
import VueNativeSock from 'vue-native-websocket'

const ws_scheme = window.location.protocol === "https:" ? "wss" : "ws";
const ws_path = ws_scheme + '://' + window.location.host + "/chat/stream/";
Vue.use(VueNativeSock, ws_path, { format: 'json', reconnection: true })
