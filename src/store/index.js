import Vue from 'vue';
import Vuex from 'vuex';


Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    darkMode: true,
    useNotification: false,
    useNotificationSound: true,
    listChannels: [],
    dictChannels: {},
  },
  getters: {
    darkMode: state => {
      return state.darkMode
    },
    useNotification: state => {
      return state.useNotification
    },
    useNotificationSound: state => {
      return state.useNotificationSound
    },
    listChannels: state => {
      return state.listChannels
    },
    dictChannels: state => {
      return state.dictChannels
    },
    channel: state => name => {
      for (const channel of state.listChannels) {
        if (name.toLowerCase() === channel.name.toLowerCase()) {
          return channel
        }
      }
    },
  },
  mutations: {
    darkMode: (state, val) => {
      if (typeof(val) === "boolean") {
        state.darkMode = val
      }
    },
    useNotification: (state, val) => {
      if (typeof(val) === "boolean") {
        state.useNotification = val
      }
    },
    useNotificationSound: (state, val) => {
      if (typeof(val) === "boolean") {
        state.useNotificationSound = val
      }
    },
    initChannels: (state, dictChannels) => {
      if (state.listChannels.length === 0) {
        for (const [name, ad, id] of Object.entries(dictChannels)) {
          let ads = []
          ads.push({
            x: 0,
            y: ad.ad})
          ads.push({
            x: 1,
            y: ad.ad})
          state.listChannels.push({'name': name, 'ads': ads, 'id': dictChannels[name].id})
          Vue.set(state.dictChannels, name, {'ads': ads, 'id': dictChannels[name].id})
        }
      }
    },
    addToChannels: (state, dictChannels) => {
      for (const channel of state.listChannels) {
        let ad = dictChannels[channel.name].ad
        let last_point = channel.ads[channel.ads.length -1]
        if (ad === last_point.y) {
          Vue.set(channel.ads, channel.ads.length -1, {x: last_point.x + 1, y: ad})
        } else {
          channel.ads.push({x: last_point.x + 1, y: ad})
        }
        if (channel.ads.length > 20000) {
          channel.ads.shift()
        }
      }
    }
  },
});
