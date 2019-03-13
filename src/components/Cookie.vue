<template>
    <transition name="fade">
    <div class="alert"
         v-if="show"
         @click="activateAudio()"
         :style="{ 'background-color': backgroundColor }">
        <h1 class="font">
            Tap for<br>Sound<br>
            <i class="material-icons icon">volume_up</i>
        </h1>
    </div>
    </transition>
</template>

<script>
  import {mapMutations} from 'vuex'

  export default {
    name: "Cookie",
    data() {
        return {
            cookieName: 'settings',
            iOS : !!navigator.platform && /iPad|iPhone|iPod/.test(navigator.platform)
        }
    },
    computed: {
      settings() {
        return [this.$store.getters.darkMode, this.$store.getters.useNotification, this.$store.getters.useNotificationSound]
      },
      show() {
          return !this.$store.getters.iosLoaded &&
                  this.iOS &&
                  this.$store.getters.useNotification &&
                  this.$store.getters.useNotificationSound &&
                  this.$route.path !== '/'
      },
      backgroundColor() {
          return this.$store.getters.darkMode ? '#414141' : '#f8f8f8'
      }
    },
    methods: {
      ...mapMutations([
        'darkMode',
        'useNotification',
        'useNotificationSound',
        'audio'
      ]),
      activateAudio() {
        this.$store.commit('audio')
      }
    },
    mounted() {
      let cookie = this.$cookie.get(this.cookieName)
      if (cookie != null) {
        cookie = JSON.parse(cookie)
        this.$store.commit('darkMode', cookie.darkMode)
        this.$store.commit('useNotification', cookie.useNotification)
        this.$store.commit('useNotificationSound', cookie.useNotificationSound)
      }
    },
    watch: {
      settings() {
        let cookie = this.$cookie.get(this.cookieName)
        if (cookie === null) {
          cookie = {}
        } else {
          cookie = JSON.parse(cookie)
        }
        cookie['darkMode'] = this.$store.getters.darkMode
        cookie['useNotification'] = this.$store.getters.useNotification
        cookie['useNotificationSound'] = this.$store.getters.useNotificationSound
        this.$cookie.set(this.cookieName, JSON.stringify(cookie), {expires: '1Y'})
      },
    },
  }
</script>

<style scoped>
.alert {
    position: fixed;
    display: flex;
    top: 50%;
    left: 50%;
    height: 80%;
    width: 80%;
    opacity: 0.9;
    border-radius: 5px;
    justify-content: center;
    align-items: center;
    transform: translate(-50%, -50%);
    z-index: 999999;
}
.font {
    font-size: 50px;
    text-align: center;
    opacity: 1;
    font-weight: lighter;
}
.icon {
    font-size: 50px;
}
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.8s;
}
.fade-enter, .fade-leave-to /* .fade-leave-active below version 2.1.8 */ {
  opacity: 0;
}
</style>
