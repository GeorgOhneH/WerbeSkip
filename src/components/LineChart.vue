<script>
  import {Line, mixins} from 'vue-chartjs'

  export default {
    extends: Line,
    props: ['xmax', 'xmin','data', 'name'],
    methods: {
      options() {
        return {
          responsive: true,
          maintainAspectRatio: (!this.styles),
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
            enabled: false,
          },
          scales: {
            xAxes: [{
              type: 'linear',
              display: false, //this will remove all the x-axis grid lines
              ticks: {
                min: this.xmin,
                max: this.xmax,
              }
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
      this.renderChart(this.data, this.options())
    },
    watch: {
      name() {
        this.renderChart(this.data, this.options())
      },
      data() {
        let chart = this.$data._chart
        chart.data.datasets[0] = this.data.datasets[0]
        chart.options = this.options()
        chart.update()
      }
    }
  }
</script>

<style scoped>

</style>
