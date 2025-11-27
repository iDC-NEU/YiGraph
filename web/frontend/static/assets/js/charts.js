!(function (MainApp) {
    "use strict";

    var usageOverview = {
      labels : [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
        ],
        dataUnit : 'Word Used',
        lineTension : .4,
        datasets : [{
            fill: true,
            label: "Word Used",
            data: [
                260, 270, 280, 400, 550, 400, 390, 360, 400, 500, 700, 900, 800,
                600, 550, 500, 520, 540, 580, 620, 670, 700, 700, 710, 690, 620,
                640, 680, 660, 630,
            ],
            color : "rgb(37,99,235)",
            background : "rgba(37,99,235, 0.5)",
        }]
    };
    MainApp.Chart.Line = function(selector, set_data){
        let elm = document.querySelectorAll(selector);
        elm.forEach(item => {
            let _get_data = (typeof set_data === 'undefined') ? eval(item.id) : set_data;
            const selectCanvas = document.getElementById(item.id).getContext("2d");
  
            const chart_data = [];
            for (let i = 0; i < _get_data.datasets.length; i++) {
                chart_data.push({
                    label: _get_data.datasets[i].label,
                    tension: _get_data.lineTension,
                    backgroundColor: _get_data.datasets[i].background,
                    fill: true,
                    borderWidth: 2,
                    borderDash: _get_data.datasets[i].dash,
                    borderColor: _get_data.datasets[i].color,
                    pointBorderColor: "transparent",
                    pointBackgroundColor: "transparent",
                    pointHoverBackgroundColor: "#fff",
                    pointHoverBorderColor: _get_data.datasets[i].color,
                    pointBorderWidth: 2,
                    pointHoverRadius: 4,
                    pointHoverBorderWidth: 2,
                    pointRadius: 4,
                    pointHitRadius: 4,
                    data: _get_data.datasets[i].data,
                });
            } 
            const chart = new Chart(selectCanvas, {
                type: 'line',
                data: {
                    labels: _get_data.labels,
                    datasets: chart_data,
                },
                options: {
                    plugins: {
                        legend: {
                            display: false,
                        },
                        tooltip: {
                            enabled: true,
                            displayColors: false,
                            callbacks: {
                                label: function (context) {
                                    return `${context.parsed.y} ${_get_data.dataUnit}`;
                                },
                            },
                            backgroundColor: "rgb(30 41 59)",
                            borderColor: "rgb(30 41 59)",
                            borderWidth: 2,
                            titleFont:{
                                size: 13,
                            },
                            titleColor: '#6783b8',
                            titleMarginBottom: 6,
                            bodyColor: '#9eaecf',
                            bodyFont:{
                                size: 12
                            },
                            bodySpacing: 4,
                            padding: 10,
                            footerMarginTop: 0,
                            displayColors: false
                        },
                    },
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            display: true,
                            beginAtZero: true,
                            ticks: {
                                color: "rgb(100 116 139)",
                                font: {
                                    size: "11px",
                                },
                                padding: 10,
                            },
                            grid: {
                                color: "rgb(226,232,240,.5)",
                                tickLength: 0,
                            },
                        },
                        x: {
                            display: true,
                            ticks: {
                                color: "rgb(100 116 139)",
                                font: {
                                    size: "9px",
                                },
                                source: "auto",
                                padding: 5,
                                reverse: MainApp.State.isRTL
                            },
                            grid: {
                                color: "transparent",
                                tickMarkLength: 0,
                                zeroLineColor: "transparent",
                            }
                        }
                    }
                }
            });
        })
    }
  
    MainApp.Chart.init = function() {
        MainApp.Chart.Line('.line-chart');
    }
    MainApp.docReady(MainApp.Chart.init);
  
  })(MainApp);