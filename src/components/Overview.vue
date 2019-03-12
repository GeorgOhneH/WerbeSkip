<template>
  <v-container grid-list-md>
    <v-text-field
      class="mx-4"
      label="Search"
      prepend-inner-icon="search"
      v-model="search"
    ></v-text-field>
    <div class="text-xs-center mt-5" v-if="hide">
      <v-progress-circular
        indeterminate
        :size="70"
        :width="2"
      ></v-progress-circular>
    </div>
    <v-data-iterator
      :items="listChannels"
      :rows-per-page-items="rowsPerPageItems"
      rows-per-page-text="Kanäle pro Seite"
      content-tag="v-layout"
      :search="search"
      no-data-text=""
      :hide-actions="hide"
      row
      wrap
    >
      <v-flex
        slot="item"
        slot-scope="props"
        xs12
        sm6
        md4
        lg3
      >
        <div @click="activateAudio()" >
          <overview-channel :channel="props.item"></overview-channel>
        </div>
      </v-flex>
    </v-data-iterator>
  </v-container>
</template>

<script>
  import OverviewChannel from './OverviewChannel'

  import { mapGetters, mapMutations } from 'vuex'

  export default {

    components: {
      OverviewChannel
    },
    data: () => ({
        rowsPerPageItems: [4, 8, 12,],
        pagination: {
          rowsPerPage: 2
        },
        search: '',
      }
    ),
    computed: {
      ...mapGetters([
        'listChannels'
      ]),
      hide() {
        return this.listChannels.length === 0;
      }
    },
    methods: {
      ...mapMutations([
          'audio'
      ]),
      activateAudio() {
        this.$store.commit('audio')
      }
    }
  }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
