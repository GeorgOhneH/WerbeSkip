<template>
</template>

<script>
  import {mapGetters} from 'vuex'

  export default {
    name: "Notification",
    props: ['status', 'channel'],
    computed: {
      ...mapGetters([
        'useNotification',
        'useNotificationSound',
      ]),
      useNotification() {
        return this.$store.getters.useNotification
      }
    },
    mounted() {
      this.requestPermission()
    },
    methods: {
      notification() {
        if (this.$store.getters.useNotificationSound) {
          if (window.webpackHotUpdate) {
            let audio = new Audio('/static/sounds/notification.mp3');
            audio.play();
          } else {
            let audio = new Audio('/staticfiles/sounds/notification.mp3');
            audio.play();
          }
        }
        let notification = new Notification(this.channel, {
          icon: 'http://cdn.sstatic.net/stackexchange/img/logos/so/so-icon.png',
          body: "Status: " + this.status.toString(),
          vibrate: [200, 100, 200],
        });
      },
      requestPermission() {
        if (this.useNotification) {
          // Let's check if the browser supports notifications
          if (Notification.permission !== "granted") {
            Notification.requestPermission();
          }
        }
      }
    },
    watch: {
      status() {
        if (this.useNotification) {
          // Let's check if the browser supports notifications
          if (Notification.permission !== "granted") {
            Notification.requestPermission();
          }
          else {
            this.notification()
          }
        }
      },
      useNotification() {
        this.requestPermission()
      }
    }
  }
</script>

<style scoped>

</style>
