<template>

  <v-list>
    <v-list-group
      prepend-icon="tv"
    >
      <v-list-tile slot="activator">
        <v-list-tile-content>
          <v-list-tile-title>Channels</v-list-tile-title>
        </v-list-tile-content>
      </v-list-tile>

      <v-list-tile
        v-for="channel in listChannels"
        :key="channel.name"
        :to="{path: '/' + channel.name}"
        ripple
      >
        <v-list-tile-content>
          <v-list-tile-title>{{ channel.name }}</v-list-tile-title>
        </v-list-tile-content>

        <v-list-tile-action>
          <v-fade-transition>
            <v-icon v-if="channel.ads[channel.ads.length-1]" :style="{color: $color.green}">lens</v-icon>
            <v-icon v-else :style="{color: $color.red}">lens</v-icon>
          </v-fade-transition>
        </v-list-tile-action>
      </v-list-tile>

    </v-list-group>
    <v-list-group
      prepend-icon="settings"
    >
      <v-list-tile slot="activator">
        <v-list-tile-content>
          <v-list-tile-title>Settings</v-list-tile-title>
        </v-list-tile-content>
      </v-list-tile>

      <v-list-tile>
        <v-list-tile-content>
          <v-list-tile-title>Dark Mode</v-list-tile-title>
        </v-list-tile-content>

        <v-list-tile-action>
          <v-switch
            v-model="darkMode"
          ></v-switch>
        </v-list-tile-action>
      </v-list-tile>
    </v-list-group>
  </v-list>
</template>

<script>
  import {mapGetters, mapMutations} from 'vuex'

  export default {
    name: "DrawerDialogContent",
    computed: {
      ...mapGetters([
        'darkMode',
        'listChannels'
      ]),
      darkMode: {
        get() {
          return this.$store.getters.darkMode
        },
        set(val) {
          this.$store.commit('darkMode', val)
        }
      },
      listChannels: {
        get() {
          return this.$store.getters.listChannels
        },
        set(val) {
        }
      },
    },
    methods: {
      ...mapMutations([
        'darkMode',
        'listChannels'
      ]),
    },
    watch: {
      '$route'(to, from, next) {
        this.$emit('close')
        next()
      }
    }
  }
</script>

<style scoped>

</style>