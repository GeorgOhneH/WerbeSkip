import Vue from 'vue';
import Vuex from 'vuex';


Vue.use(Vuex)

function init_audio() {
  if (window.webpackHotUpdate) {
    return new Audio('/static/sounds/notification.mp3');
  } else {
    return new Audio('/staticfiles/sounds/notification.mp3');
  }

}


export default new Vuex.Store({
  state: {
    darkMode: true,
    useNotification: false,
    useNotificationSound: true,
    listChannels: [],
    dictChannels: {},
    audio: init_audio(),
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
    audio: state => {
      return state.audio
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
    audio: (state) => {
        state.audio = init_audio()
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
          console.log(dictChannels)
        for (const [name, data] of Object.entries(dictChannels)) {
          let ads = []
          for (const data_point of data.ads) {
            ads.push({
              x: data_point.x,
              y: data_point.ad})
          }

          state.listChannels.push({'name': name, 'ads': ads, 'id': data.id})
          Vue.set(state.dictChannels, name, {'ads': ads, 'id': data.id})
        }
      }
    },
    addToChannels: (state, dictChannels) => {
      for (const channel of state.listChannels) {
        let ad = dictChannels[channel.name].ad
        if (channel.ads.length === 0) {
          Vue.set(channel.ads, 0, {x: 1, y: ad})
        } else {
          let last_point = channel.ads[channel.ads.length - 1]
          if (ad === last_point.y) {
            Vue.set(channel.ads, channel.ads.length - 1, {x: last_point.x + 1, y: ad})
          } else {
            channel.ads.push({x: last_point.x + 1, y: ad})
          }
          if (channel.ads.length > 2000) {
            channel.ads.shift()
          }
        }
      }
    }
  },
});
