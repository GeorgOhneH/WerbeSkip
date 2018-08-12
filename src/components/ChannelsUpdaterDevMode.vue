<template>
  <v-btn v-if="use" @click="updateChannels()">Update</v-btn>
</template>

<script>
  import {mapMutations} from 'vuex'


  export default {
    name: "ChannelsUpdater",
    props: ['use'],
    data() {
      return {
        interval: undefined,
      }
    },
    watch: {
      use: function () {
        if (this.use) {
          this.$store.commit('initChannels', this.getRawChannels())
          this.interval = setInterval(this.updateChannels, 1000)
        }
      }
    },
    beforeDestroy() {
      if (this.interval) {
        clearInterval(this.interval)
        this.interval = undefined
      }
    },
    methods: {
      ...mapMutations([
        'initChannels',
        'addToChannels'
      ]),
      updateChannels() {
        this.$store.commit('addToChannels', this.getRawChannels())
      },
      getRawChannels() {
        return {
          'Prosieben':
            {
              'ad': Math.random() >= 0.5
            },
          'SAT':
            {
              'ad': Math.random() >= 0.5
            },
          'SRF':
            {
              'ad': Math.random() >= 0.5
            },
          'RTL':
            {
              'ad': Math.random() >= 0.5
            },
          'ZDF':
            {
              'ad': Math.random() >= 0.5
            },
          'Twitch':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel200':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel201':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel202':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel203':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel204':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel205':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel206':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel207':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel208':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel209':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel210':
            {
              'ad': Math.random() >= 0.5
            },
          'Channel211':
            {
              'ad': Math.random() >= 0.5
            },
        }
      }
    }
  }
</script>

<style scoped>

</style>
