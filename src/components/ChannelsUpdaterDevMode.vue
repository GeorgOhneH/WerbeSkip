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
          this.$store.commit('initChannels', this.getRawChannelsInit())
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
      getRawChannelsInit() {
        return {
          'Prosieben':
            [{
              'ad': Math.random() >= 0.5,
              'x': -100,
              'id': 354
            },{
              'ad': Math.random() >= 0.5,
              'x': -50,
              'id': 354
            },{
              'ad': Math.random() >= 0.5,
              'x': -10,
              'id': 354
            }],
        }
      },
      getRawChannels() {
        return {
          'Prosieben':
            {
              'ad': Math.random() >= 0.5,
              'id': 354
            },
        }
      },
    }
  }
</script>

<style scoped>

</style>
