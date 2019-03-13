<template>
  <v-navigation-drawer
    app
    v-model="open"
    clipped
    disable-resize-watcher
  >
    <drawer-dialog-content></drawer-dialog-content>
  </v-navigation-drawer>
</template>

<script>
  import DrawerDialogContent from './DrawerDialogContent'

  export default {
    name: "DrawerDialog",
    props: ['value'],
    components: {
      DrawerDialogContent,
    },
    computed: {
      open: {
        get() {
          return this.value
        },
        set(val) {
          this.$emit('input', val)
        }
      }
    },
  }
</script>

<style scoped>

</style>
