import Vue from 'vue';
import Vuex from 'vuex';


Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    darkMode: true,
    listChannels: [],
    dictChannels: {},
  },
  getters: {
    darkMode: state => {
      return state.darkMode
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
    initChannels: (state, dictChannels) => {
      if (state.listChannels.length === 0) {
        for (const [name, ad] of Object.entries(dictChannels)) {
          let ads = []
          state.listChannels.push({'name': name, 'ads': ads})
          Vue.set(state.dictChannels, name, {'ads': ads})
        }
      }
    },
    addToChannels: (state, dictChannels) => {
      for (const channel of state.listChannels) {
        channel.ads.push(dictChannels[channel.name].ad)
        if (channel.ads.length > 1500) {
          channel.ads.shift()
        }
      }
    }
  },
});
