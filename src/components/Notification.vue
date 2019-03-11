<template>
</template>

<script>
  import {mapGetters} from 'vuex'

  export default {
    name: "Notification",
    props: ['status', 'channel', 'id'],
    computed: {
      ...mapGetters([
        'useNotification',
        'useNotificationSound',
        'audio',
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
          this.$store.audio.play();
        }
        if (Notification.permission !== "granted") {
          Notification.requestPermission();
        } else {
          let notification = new Notification(this.channel, {
            icon: 'https://media.cinergy.ch/t_station/' + this.id + '/icon160_light.png',
            body: "Status: " + this.status.toString(),
            vibrate: [200, 100, 200],
          });
        }
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
            this.notification()
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
