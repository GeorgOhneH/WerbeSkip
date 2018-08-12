<template>
</template>

<script>
  import {mapMutations} from 'vuex'


  export default {
    name: "ChannelsUpdater",
    created() {
      this.$options.sockets.onmessage = (message) => this.messageReceived(message.data)
    },
    methods: {
      ...mapMutations([
        'initChannels',
        'addToChannels'
      ]),
      messageReceived(data) {
        data = JSON.parse(data)
        console.log(data.channel)
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
