<template>
  <v-app :dark="darkMode">
    <Cookie></Cookie>
    <drawer-dialog v-model="drawerDialog"></drawer-dialog>
    <v-toolbar
      app
      clipped-left
    >
      <v-toolbar-side-icon @click.stop="drawerDialog = !drawerDialog"></v-toolbar-side-icon>
      <router-link :to="{path: '/'}" tag="div">
        <v-btn flat round class="text-normal">
          <v-toolbar-title>{{title}}</v-toolbar-title>
        </v-btn>
      </router-link>

      <v-spacer></v-spacer>
      <channels-updater></channels-updater>
    </v-toolbar>
    <v-content>
      <router-view></router-view>
    </v-content>
  </v-app>
</template>

<script>
  import {mapGetters} from 'vuex'

  import Overview from './components/Overview'
  import DrawerDialog from './components/DrawerDialog'
  import ChannelsUpdater from './components/ChannelsUpdater'
  import Cookie from './components/Cookie'

  export default {
    name: 'App',
    components: {
      Overview,
      DrawerDialog,
      Cookie,
      ChannelsUpdater
    },
    data() {
      return {
        title: 'WerbeSkip',
        drawerDialog: false,
      }
    },
    computed: {
      ...mapGetters([
        'darkMode'
      ]),
      darkMode() {
        return this.$store.getters.darkMode
      }
    },
  }
</script>

<style scoped>
  .text-normal {
    text-transform: none !important;
  }
</style>
