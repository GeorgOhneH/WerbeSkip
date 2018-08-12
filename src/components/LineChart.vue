<script>
  import {Line, mixins} from 'vue-chartjs'

  export default {
    extends: Line,
    mixins: [mixins.reactiveProp],
    data() {
      return {
        options: {
          responsive: true,
          maintainAspectRatio: true,
          animation: {
            duration: 1000,
            easing: 'linear'
          },
          elements: {
            point: {
              radius: 0
            }
          },
          legend: {
            display: false
          },
          tooltips: {
            enabled: false
          },
          scales: {
            xAxes: [{
              display: false, //this will remove all the x-axis grid lines
            }],
            yAxes: [{
              display: false, //this will remove all the x-axis grid lines
              ticks: {
                max: 1.1,
                min: -0.1,
              }
            }]
          }
        }
      }
    },
    mounted() {
      if (this.styles) {
        this.options.maintainAspectRatio = false
      }
      this.renderChart(this.chartData, this.options)
    },
  }
</script>

<style scoped>

</style>