import Vue from 'vue'
import VueRouter from 'vue-router'

import Overview from '../components/Overview'
import Channel from '../components/Channel'

Vue.use(VueRouter)

export default new VueRouter({
  mode: 'history',
  routes: [
    {path: '/', component: Overview},
    {path: '/:channel', component: Channel}
  ]
})