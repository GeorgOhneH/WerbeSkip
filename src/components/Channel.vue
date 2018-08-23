<template>
  <v-container>
    <div v-if="loading" class="text-xs-center mt-5">
      <v-progress-circular
        indeterminate
        :size="70"
        :width="2"
      ></v-progress-circular>
    </div>
    <div v-else-if="error404">
      <error404></error404>
    </div>
    <div v-else>
      <notification :status="ad" :channel="name"></notification>
      <h1 v-if="$vuetify.breakpoint.smAndUp" class="display-4">{{name}}</h1>
      <h1 v-else class="display-3" style="font-weight: 300">{{name}}</h1>

      <v-divider class="mb-4"></v-divider>
      <h1 class="display-1" style="font-weight: 300;">Status</h1>

      <h1 v-if="$vuetify.breakpoint.smAndUp" class="display-4">{{ad}}</h1>
      <h1 v-else class="display-3" style="font-weight: 300">{{ad}}</h1>
      <v-divider class="my-4"></v-divider>

      <h1 class="display-1 mt-3" style="font-weight: 300;">History</h1>
      <chart :ads="channel.ads" :styles="{height: height, position: 'relative'}"></chart>

    </div>
  </v-container>
</template>

<script>
  import Error404 from './Error404'
  import Chart from './Chart'
  import Notification from './Notification'

  import {mapGetters} from 'vuex'

  export default {
    name: "Channel",
    components: {
      Error404,
      Chart,
      Notification,
    },
    computed: {
      ...mapGetters([
        'channel',
        'listChannels'
      ]),
      urlName() {
        return this.$route.params.channel
      },
      channel() {
        return this.$store.getters.channel(this.urlName)
      },
      name() {
        if (this.channel === undefined) {
          return ''
        }
        return this.channel.name
      },
      loading() {
        return this.$store.getters.listChannels.length === 0
      },
      error404() {
        return this.channel === undefined && !this.loading;
      },
      ad() {
        if (this.channel.ads[this.channel.ads.length - 1]) {
          return 'No Ads'
        }
        return 'Ads'
      },
      height() {
        if (this.$vuetify.breakpoint.lgAndUp) {
          return '500px'
        } else if (this.$vuetify.breakpoint.mdOnly) {
          return '400px'
        } else if (this.$vuetify.breakpoint.smOnly) {
          return '300px'
        } else if (this.$vuetify.breakpoint.xsOnly) {
          return '200px'
        }
      },
    },
  }
</script>

<style scoped>

</style>
