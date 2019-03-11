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

      <div v-if="loading" class="text-xs-center my-3">
        <v-progress-circular
          indeterminate
          :size="30"
          :width="2"
        ></v-progress-circular>
      </div>

      <v-list-tile
        v-else
        v-for="channel in listChannels"
        :key="channel.name"
        :to="{path: '/' + channel.name.toLowerCase()}"
        ripple
      >
        <v-list-tile-content>
          <v-list-tile-title>{{ channel.name }}</v-list-tile-title>
        </v-list-tile-content>

        <v-list-tile-action>
          <v-fade-transition>
            <v-icon v-if="channel.ads[channel.ads.length-1].y" :style="{color: $color.green}">lens</v-icon>
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

      <v-list-tile>
        <v-list-tile-content>
          <v-list-tile-title>Notification</v-list-tile-title>
        </v-list-tile-content>
          <div @click="activateAudio()">
            <v-list-tile-action>
              <v-switch
                v-model="useNotification"
              ></v-switch>
            </v-list-tile-action>
        </div>
      </v-list-tile>

      <v-list-tile>
        <v-list-tile-content>
          <v-list-tile-title>Notification Sound</v-list-tile-title>
        </v-list-tile-content>
          <div @click="activateAudio()">
            <v-list-tile-action :class="{'switch-disable': !useNotification}">
              <v-switch
                v-model="useNotificationSound"
                :disabled="!useNotification"
              ></v-switch>
            </v-list-tile-action>
          </div>
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
        'useNotification',
        'useNotificationSound',
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
      useNotification: {
        get() {
          return this.$store.getters.useNotification
        },
        set(val) {
          this.$store.commit('useNotification', val)
        }
      },
      useNotificationSound: {
        get() {
          return this.$store.getters.useNotificationSound
        },
        set(val) {
          this.$store.commit('useNotificationSound', val)
        }
      },
      listChannels: {
        get() {
          return this.$store.getters.listChannels
        },
        set(val) {
        }
      },
      loading() {
        return this.listChannels.length === 0
      }
    },
    methods: {
      ...mapMutations([
        'darkMode',
        'useNotification',
        'useNotificationSound',
        'listChannels',
        'audio',
      ]),
      activateAudio() {
        this.$store.commit('audio')
      }
    },
    watch: {
      $route(to, from) {
        this.$emit('close')
      }
    },
  }
</script>

<style scoped>
.switch-disable {
  cursor: not-allowed;
}
</style>
