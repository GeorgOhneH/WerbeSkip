import Vue from 'vue'
import './plugins/vuetify'
import './plugins/costume'
import './plugins/websocket'

import store from './store'
import router from './router'

import App from './App.vue'

Vue.config.productionTip = false

new Vue({
  store,
  router,
  render: h => h(App),
}).$mount('#app')

