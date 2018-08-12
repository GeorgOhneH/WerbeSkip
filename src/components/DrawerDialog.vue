<template>
  <v-navigation-drawer
    v-if="$vuetify.breakpoint.mdAndUp"
    app
    :value="open"
    clipped
  >
    <drawer-dialog-content @close="open = false"></drawer-dialog-content>
  </v-navigation-drawer>

  <v-dialog
    app
    v-else
    v-model="open"
    fullscreen
    hide-overlay
    transition="dialog-bottom-transition"
  >
    <v-card>
      <v-toolbar dark color="primary">
        <v-toolbar-title class="ml-3">WerbeSkip</v-toolbar-title>
        <v-spacer></v-spacer>
        <v-btn icon dark @click.native="open = false">
          <v-icon>close</v-icon>
        </v-btn>
      </v-toolbar>
      <drawer-dialog-content @close="open = false"></drawer-dialog-content>
    </v-card>
  </v-dialog>
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
