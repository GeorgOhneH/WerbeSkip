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
    },
    methods: {
      notification() {
        if (this.$store.getters.useNotificationSound) {
          var audio = new Audio('/staticfiles/sounds/notification.mp3');
          audio.play();
        }
        var notification = new Notification(this.channel, {
          icon: 'http://cdn.sstatic.net/stackexchange/img/logos/so/so-icon.png',
          body: "Status: " + this.status.toString(),
          vibrate: [200, 100, 200],
        });
      }
    },
    watch: {
      status() {
        if (this.$store.getters.useNotification) {
          // Let's check if the browser supports notifications
          if (Notification.permission !== "granted")
            Notification.requestPermission();
          else {
            this.notification()
          }
        }
      }
    }
  }
</script>

<style scoped>

</style>
