<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <button v-on:click="clickButton">test</button>
  </div>
</template>

<script>
  export default {
    data() {
      return {
        // note: changing this line won't causes changes
        // with hot-reload because the reloaded component
        // preserves its current state and we are modifying
        // its initial state.
        msg: 'Foo!'
      }
    },

    methods: {
      clickButton: function (val) {
        // $socket is [WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket) instance
        // or with {format: 'json'} enabled
        this.$socket.sendObj({"command": 'update', "room": 1, "channel": "testtest"})
        console.log("send message")
      }
    },
    mounted() {
      this.$options.sockets.onmessage = (data) => console.log(data)
    }
  }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  h1 {
    color: #42b983;
  }
</style>
