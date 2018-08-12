<template>
  <div ref="container" v-resize="onResize">
    <line-chart
      :chart-data="datacollection"
      :styles="styles"
    ></line-chart>
  </div>
</template>

<script>
  import LineChart from './LineChart'

  export default {
    name: "Chart",
    components: {
      LineChart
    },
    props: ['ads', 'styles'],
    data() {
      return {
        limit: 0,
        croppedAds: [],
        bluePrint: {
          labels: [],
          datasets: [
            {
              borderColor: '#3cba54',
              steppedLine: true,
              fill: false,
              data: [],
              borderWidth: 1.5,
            }
          ]
        }
      }
    },
    computed: {
      datacollection() {
        let datacollection = Object.assign({}, this.bluePrint)

        datacollection.labels = this.croppedAds
        datacollection.datasets[0].data = this.croppedAds

        if (this.croppedAds[this.croppedAds.length - 1]) {
          datacollection.datasets[0].borderColor = this.$color.green
        }
        else {
          datacollection.datasets[0].borderColor = this.$color.red
        }
        return datacollection
      },
    },
    mounted() {
      this.onResize()
    },
    methods: {
      onResize() {
        this.setLimit(this.$refs.container.clientWidth)
      },
      setLimit(newLimit) {
        this.limit = newLimit
        this.setAds()
      },
      setAds() {
        this.croppedAds = this.ads.slice(-this.limit)
      }
    },
    watch: {
      ads() {
        this.croppedAds.push(this.ads[this.ads.length-1])
        if (this.croppedAds.length > this.limit) {
          this.croppedAds.shift()
        }
      }
    }
  }
</script>

<style scoped>

</style>