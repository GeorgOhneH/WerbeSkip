<template>
  <v-navigation-drawer
    v-if="$vuetify.breakpoint.mdAndUp"
    app
    :value="value"
    clipped
  >
    <drawer-dialog-content @close="value = false"></drawer-dialog-content>
  </v-navigation-drawer>

  <v-dialog
    app
    v-else
    v-model="value"
    fullscreen
    hide-overlay
    transition="dialog-bottom-transition"
  >
    <v-card>
      <v-toolbar dark color="primary">
        <v-toolbar-title class="ml-3">WerbeSkip</v-toolbar-title>
        <v-spacer></v-spacer>
        <v-btn icon dark @click.native="value = false">
          <v-icon>close</v-icon>
        </v-btn>
      </v-toolbar>
      <drawer-dialog-content @close="value = false"></drawer-dialog-content>
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
    watch: {
      value: function (val) {
        this.$emit('input', val)
      }
    }
  }
</script>

<style scoped>

</style>