// Shifting eigenvalues
function shift_eig(eig, lr, beta){
  var discriminant =  math.subtract(math.square(math.subtract(1+beta, math.multiply(lr, eig))), 4*beta);

  var new_eig1 = math.multiply(0.5, (math.add(math.subtract(1+beta, math.multiply(lr, eig)), math.sqrt(discriminant))));

  var new_eig2 = math.multiply(0.5, (math.subtract(math.subtract(1+beta, math.multiply(lr, eig)), math.sqrt(discriminant))));

  return [new_eig1, new_eig2];
}

function initialize_chart(class_name, pixels, units) {
  var b = {}; // chart-related handlers and objects

  b.pixels = pixels || 600;
  b.units = units || 2;
  b.unit = b.pixels / b.units / 2; // unit in pixels
  b.class_name = class_name;
  b.chart = d3.select('.' + b.class_name)
  .attr('width', b.pixels)
  .attr('height', b.pixels);

  // create scales
  b.xScale = d3.scaleLinear()
                           .domain([-b.units,b.units])
                           .range([0,b.pixels]);
  b.yScale = d3.scaleLinear()
                           .domain([b.units,-b.units])
                           .range([0,b.pixels]);
  b.xRevScale = d3.scaleLinear()
                          .domain([0, b.pixels])
                          .range([-b.units, b.units]);
  b.yRevScale = d3.scaleLinear()
                          .domain([0, b.pixels])
                          .range([b.units, -b.units]);
  return b;
}

function replot(eigs, lr, beta, blob) {
  console.log('Replotting', lr, beta);

  var b = blob;
  var width = b.pixels;
  var height = b.pixels;

  // Set up axes
  var xAxis = d3.axisBottom(b.xScale);
  var xAxisGroup = b.chart.selectAll('.axis_x')
  .data([0]);
  xAxisGroup
  .enter()
  .append('g')
  .attr('class', 'axis_x')
  .attr('transform', 'translate(0, '+ height/2 + ')')
  .call(xAxis);
  xAxisGroup
  .attr('transform', 'translate(0, '+ height/2 + ')')
  .call(xAxis);

  var yAxis = d3.axisLeft(b.yScale);
  var yAxisGroup = b.chart.selectAll('.axis_y')
  .data([0]);
  yAxisGroup
  .enter()
  .append('g')
  .attr('class', 'axis_y')
  .attr('transform', 'translate(' + width/2 + ',0)')
  .call(yAxis);
  yAxisGroup
  .attr('transform', 'translate(' + width/2 + ',0)')
  .call(yAxis);

  // Print unit circle
  var unitCircle = b.chart.selectAll('.unitCircle')
  .data([0]);
  unitCircle.enter().append('circle')
  .attr('class', 'unitCircle')
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", b.unit);
  unitCircle
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", b.unit);

  var lr_box = b.chart.selectAll('.lr').data([0]);
  lr_box.enter().append('text')
  .attr('class', 'lr box')
  .attr("x", 20)
  .attr("y", 20)
  .text('lr=0.');
  lr_box
  .attr("x", 20)
  .attr("y", 20)
  .text('lr=0.');

  var beta_box = b.chart.selectAll('.beta').data([0]);
  beta_box.enter().append('text')
  .attr('class', 'beta box')
  .attr("x", 20)
  .attr("y", 40)
  .text('beta=0.');
  beta_box
  .attr("x", 20)
  .attr("y", 40)
  .text('beta=0.');

  var one = math.complex(1);
  var transition_eigs_direction = [];
  var transition_eigs = [];
  var shifted_eigs = [];
  var radius_eigs = 0.
  var radius_shifted_eigs = 0.

  // text to show value of beta and lr
  lr_box.text('lr='+lr.toFixed(2));
  beta_box.text('beta='+beta.toFixed(2))

  //////// begin trace //////////
  var eigs_traces = [];
  var shifted_eigs_traces = [];
  var lr_max = 2.;
  var lr_steps = 50;
  var lr_resolution = lr_max / lr_steps;
  // sweep learning rate up to 2.
  for(var j=0; j<lr_steps; j++) {
    var sweep_lr = j * lr_resolution;
    for(var i=0;i<eigs.length;i++){
      var new_eigs = shift_eig(eigs[i], sweep_lr, beta);

      eigs_traces[eigs_traces.length] = {
        lr: sweep_lr,
        eig: eigs[i]
      };
      shifted_eigs_traces[shifted_eigs_traces.length] = {
        lr: sweep_lr,
        eig: new_eigs[0]
      };
      shifted_eigs_traces[shifted_eigs_traces.length] = {
        lr: sweep_lr,
        eig: new_eigs[1]
      };
    }
  }

  var eigenValueTraces = b.chart.selectAll('.eigenValueTrace')
  .data(eigs_traces);

  eigenValueTraces
  .enter()
  .append("circle")
  .attr('class', 'eigenValueTrace')
  .attr("cx", function(d){return b.xScale(1-d.lr*d.eig.re);})
  .attr("cy", function(d){return b.xScale(-d.lr*d.eig.im);})
  .attr("r", 0.04*b.unit);

  eigenValueTraces
  //.transition()
  .attr("cx", function(d){return b.xScale(1-d.lr*d.eig.re);})
  .attr("cy", function(d){return b.xScale(-d.lr*d.eig.im);})
  .attr("r", 0.04*b.unit);

  eigenValueTraces
  .exit()
  .remove();


  var shiftedEigenValueTraces = b.chart.selectAll('.shiftedEigenValueTrace')
  .data(shifted_eigs_traces);

  shiftedEigenValueTraces
  .enter()
  .append("circle")
  .attr('class', 'shiftedEigenValueTrace')
  .attr("cx", function(d){return b.xScale(d.eig.re);})
  .attr("cy", function(d){return b.xScale(d.eig.im);})
  .attr("r", 0.04*b.unit);

  shiftedEigenValueTraces
  //.transition()
  .attr("cx", function(d){return b.xScale(d.eig.re);})
  .attr("cy", function(d){return b.xScale(d.eig.im);})
  .attr("r", 0.04*b.unit);

  shiftedEigenValueTraces
  .exit()
  .remove();
  //////// end trace //////////


  for(var i=0;i<eigs.length;i++){
    var new_eigs = shift_eig(eigs[i], lr, beta);
    shifted_eigs[shifted_eigs.length] = new_eigs[0];
    shifted_eigs[shifted_eigs.length] = new_eigs[1];

    var transition_eig = math.subtract(1, math.multiply(lr, eigs[i]));

    // Get convergence circles
    var abs_eig = math.abs(transition_eig);
    var abs_shifted_eig = math.max(math.abs(shifted_eigs[0]), math.abs(shifted_eigs[1]));
    radius_eigs = math.max(radius_eigs, abs_eig);
    radius_shifted_eigs = math.max(radius_shifted_eigs, abs_shifted_eig);
  }

  // Find active eigenvalues (the argmax of convergence rate)
  var radius_tolerance = 0.01;
  var active_eig = [];
  var active_shifted_eig = [];
  for(var i=0;i<eigs.length;i++){
    active_eig[i] = (math.abs(eigs[i]) >= radius_eigs - radius_tolerance);
    active_shifted_eig[i*2] = (math.abs(shifted_eigs[i*2]) >= radius_shifted_eigs - radius_tolerance);
    active_shifted_eig[i*2+1] = (math.abs(shifted_eigs[i*2+1]) >= radius_shifted_eigs - radius_tolerance);
  }


  console.log('radius' + radius_eigs);

  // Draw convergence circle
  var convergenceCircle = b.chart
  .selectAll('.convergenceCircle')
  .data([1]);

  convergenceCircle
  .enter()
  .append('circle')
  .attr('class', 'convergenceCircle')
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", radius_eigs*b.unit);

  convergenceCircle
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", radius_eigs*b.unit);

  // Draw shifted convergence circle
  var shiftedConvergenceCircle = b.chart
  .selectAll('.shiftedConvergenceCircle')
  .data([1]);

  shiftedConvergenceCircle
  .enter()
  .append('circle')
  .attr('class', 'shiftedConvergenceCircle')
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", radius_shifted_eigs*b.unit);

  shiftedConvergenceCircle
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", radius_shifted_eigs*b.unit);

  // Draw eigenvalues
  var eigenValues = b.chart.selectAll('.eigenValue')
  .data(eigs);

  console.log(eigs);
  console.log('Actual lr ' + lr);

  function active_stroke(ref) {
    // this is not functional yet
    var closure = function(d,i) {
      if(ref[i]) {
        return 2;
      } else {
        return 2;
      }
    }
    return closure;
  }

  eigenValues
  .enter()
  .append("circle")
  .attr('class', 'eigenValue')
  .attr("cx", function(d){return b.xScale(1-lr*d.re);})
  .attr("cy", function(d){return b.xScale(-lr*d.im);})
  .attr("r", 0.04*b.unit)
  .style('stroke-width', active_stroke(active_eig));

  eigenValues
  //.transition()
  .attr("cx", function(d){return b.xScale(1-lr*d.re);})
  .attr("cy", function(d){return b.xScale(-lr*d.im);})
  .attr("r", 0.04*b.unit)
  .style('stroke-width', active_stroke(active_eig));



  eigenValues
  .exit()
  .remove();



  // Draw shifted eigenvalues
  var shiftedEigenValues = b.chart.selectAll('.shiftedEigenValue')
  .data(shifted_eigs);

  shiftedEigenValues
  .enter()
  .append("circle")
  .attr('class', 'shiftedEigenValue')
  .attr("cx", function(d){return b.xScale(d.re);})
  .attr("cy", function(d){return b.xScale(d.im);})
  .attr("r", 0.04*b.unit)
  .style('stroke', active_stroke(active_shifted_eig));

  shiftedEigenValues
  //.transition()
  .attr("cx", function(d){return b.xScale(d.re);})
  .attr("cy", function(d){return b.xScale(d.im);})
  .attr("r", 0.04*b.unit)
  .style('stroke', active_stroke(active_shifted_eig));

  shiftedEigenValues
  .exit()
  .remove();


}
