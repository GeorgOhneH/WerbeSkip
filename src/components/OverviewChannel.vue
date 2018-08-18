<template>
  <router-link tag="div" :to="{path: '/' + channel.name.toLowerCase()}">
    <v-card style="cursor: pointer;">
      <v-card-title><h4 class="main-card-title">{{ channel.name }}</h4></v-card-title>
      <v-divider></v-divider>

      <v-card-text style="position: relative">
        <v-container fill-height fluid>
          <v-layout fill-height>
            <v-flex xs12 flexbox>
              <div class="over-element">
                <h2 class="ads px-4 py-2">{{ title }}</h2>
              </div>
              <chart :ads="channel.ads"></chart>
            </v-flex>
          </v-layout>
        </v-container>
      </v-card-text>
    </v-card>

  </router-link>
</template>

<script>
  import Chart from './Chart'

  export default {
    name: "OverviewChannel",
    components: {
      Chart
    },
    props: ['channel'],
    computed: {
      title() {
        if (this.channel.ads[this.channel.ads.length - 1]) {
          return 'No Ads'
        }
        else {
          return 'Ads'
        }
      },
    },
  }

</script>

<style scoped>
  .main-card-title {
    font-size: 1.5em;
    margin: auto;
    font-weight: 400;
  }

  .over-element {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
  }

  .ads {
    background-color: rgba(200, 200, 200, .25);
    font-size: 4.5em;
    line-height: 0.9;
    text-align: center;
    font-weight: 300;
  }
</style>
