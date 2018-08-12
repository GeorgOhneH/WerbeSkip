<template>
  <channels-updater-dev-mode :use="devMode"></channels-updater-dev-mode>
</template>

<script>
  import {mapMutations} from 'vuex'

  import ChannelsUpdaterDevMode from './ChannelsUpdaterDevMode'

  export default {
    name: "ChannelsUpdater",
    components: {
      ChannelsUpdaterDevMode,
    },
    data() {
      return {
        devMode: false
      }
    },
    mounted() {
      if (window.webpackHotUpdate) {
        this.devMode = true
      } else {
        this.$options.sockets.onmessage = (message) => this.messageReceived(message.data)
      }
    },
    methods: {
      ...mapMutations([
        'initChannels',
        'addToChannels'
      ]),
      messageReceived(data) {
        data = JSON.parse(data)
        if (data.channel !== undefined) {
          this.updateChannels(data.channel)
        }
      },
      updateChannels(getRawChannels) {
        this.$store.commit('initChannels', getRawChannels)
        this.$store.commit('addToChannels', getRawChannels)
      },
    }
  }
</script>

<style scoped>

</style>
