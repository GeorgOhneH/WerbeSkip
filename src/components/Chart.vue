<template>
  <div ref="container">
    <line-chart
      :data="datacollection"
      :styles="styles"
      :xmax="xmax"
      :xmin="xmin"
      :name="name"
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
    props: ['ads', 'styles', 'name'],
    data() {
      return {
        limit: 1,
        croppedAds: [],
        bluePrint: {
          datasets: [
            {
              borderColor: (this.ads.length === 0 || this.ads[this.ads.length - 1].y) ? this.$color.green : this.$color.red,
              steppedLine: 'after',
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
        if (this.croppedAds.length !== 0) {
          let datacollection = Object.assign({}, this.bluePrint)

          datacollection.datasets[0].data = this.croppedAds

          if (this.croppedAds[this.croppedAds.length - 1].y) {
            datacollection.datasets[0].borderColor = this.$color.green
          }
          else {
            datacollection.datasets[0].borderColor = this.$color.red
          }
          return datacollection
        }
        return this.bluePrint
      },
      xmax() {
        if (this.croppedAds.length !== 0) {
          return this.croppedAds[this.croppedAds.length - 1].x
        } else {
          return 2
        }
      },
      xmin() {
        if (this.croppedAds.length !== 0) {
          return this.croppedAds[0].x + 10
        } else {
          return 0
        }
      },
    },
    mounted() {
      this.onMounted()
    },
    methods: {
      onMounted() {
        this.setLimit(this.$refs.container.clientWidth * 10)
      },
      setLimit(newLimit) {
        this.limit = newLimit
        this.setAds()
      },
      setAds() {
        this.croppedAds = this.ads.slice()
        if (this.croppedAds.length !== 0) {
          this.limitArray()
        }
      },
      limitArray() {
        for (const i in [...Array(this.croppedAds.length).keys()]) {
          let current_i = this.croppedAds.length - 1 - i
          let current_obj = this.croppedAds[current_i]
          let last_obj = this.croppedAds[this.croppedAds.length - 1]
          if (last_obj.x - current_obj.x > this.limit) {
            this.$set(this.croppedAds, current_i, {x: last_obj.x - this.limit, y: current_obj.y})
            for (const y in [...Array(current_i).keys()]) {
              this.croppedAds.shift()
            }
            break
          }
        }
      }
    },
    watch: {
      name() {
        this.setAds()
      },
      ads() {
        if (this.croppedAds.length !== 0) {
          if (this.croppedAds[this.croppedAds.length - 1].y !== this.ads[this.ads.length - 1].y) {
            this.croppedAds.push(this.ads[this.ads.length - 1])
          } else {
            this.$set(this.croppedAds, this.croppedAds.length - 1, {
              x: this.croppedAds[this.croppedAds.length - 1].x + 1,
              y: this.ads[this.ads.length - 1].y
            })
          }
          this.limitArray()
        }
      }
    }
  }
</script>

<style scoped>

</style>
