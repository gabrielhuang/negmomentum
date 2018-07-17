// Shifting eigenvalues
function shift_eig(eig, lr, beta){
  var discriminant =  math.subtract(math.square(math.subtract(1+beta, math.multiply(lr, eig))), 4*beta);

  var new_eig1 = math.multiply(0.5, (math.add(math.subtract(1+beta, math.multiply(lr, eig)), math.sqrt(discriminant))));

  var new_eig2 = math.multiply(0.5, (math.subtract(math.subtract(1+beta, math.multiply(lr, eig)), math.sqrt(discriminant))));

  return [new_eig1, new_eig2];
}

// Create element with class '' if does not exist,
// else return existing element
function create_unique(root, type, class_name, other_classes) {
  other_classes = other_classes || [];
  var unique = root.selectAll('.' + class_name).data([0]);
  unique.enter().append(type).classed(class_name, true);
  var unique = root.selectAll('.' + class_name);
  for(var i=0; i<other_classes.length;i++) {
    unique.classed(other_classes[i], true);
  }
  return unique;
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
  var xAxisGroup = create_unique(b.chart, 'g', 'axis_x')
  .attr('transform', 'translate(0, '+ height/2 + ')')
  .call(xAxis);

  var yAxis = d3.axisLeft(b.yScale);
  var yAxisGroup = create_unique(b.chart, 'g', 'axis_y')
  .attr('transform', 'translate(' + width/2 + ',0)')
  .call(yAxis);

  // Print unit circle
  var unitCircle = create_unique(b.chart, 'circle', 'unitCircle')
  .attr("cx", height/2)
  .attr("cy", width/2)
  .attr("r", b.unit);

  var lr_box = create_unique(b.chart, 'text', 'lr', ['box'])
  .attr("x", 20)
  .attr("y", 20)
  .text('lr=0.');

  var beta_box = create_unique(b.chart, 'text', 'beta', ['box'])
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
  beta_box.text('beta='+beta.toFixed(2));

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
    var abs_shifted_eig = math.max(math.abs(new_eigs[0]), math.abs(new_eigs[1]));
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



function initialize_touchpad(class_name, pixels,
    min_lr, max_lr, min_beta, max_beta) {
  var tp = {}; // chart-related handlers and objects

  tp.pixels = pixels;
  tp.min_lr = min_lr === undefined? -2 : min_lr;
  tp.max_lr = max_lr === undefined? 2. : max_lr;
  tp.min_beta = min_beta === undefined? -2.:min_beta;
  tp.max_beta = max_beta === undefined? 2.:max_beta;
  tp.bar_thick = 10;

  tp.yScale = d3.scaleLinear()
                           .domain([tp.max_lr,tp.min_lr])
                           .range([0,tp.pixels]);
  tp.xScale = d3.scaleLinear()
                           .domain([tp.min_beta, tp.max_beta])
                           .range([0,tp.pixels]);
  tp.yRevScale = d3.scaleLinear()
                          .domain([0,tp.pixels])
                          .range([tp.max_lr,tp.min_lr]);
  tp.xRevScale = d3.scaleLinear()
                          .domain([0,tp.pixels])
                          .range([tp.min_beta, tp.max_beta]);

  tp.class_name = class_name;
  tp.touchpad = d3.select('.' + tp.class_name)
  .attr('width', tp.pixels)
  .attr('height', tp.pixels);

  return tp;
}


function sweep(eigs, lr_min, lr_max, beta_min, beta_max, resolution) {
  resolution = resolution || 10;
  //var all_radius = [];
  var shifted_rates = [];
  var flat_shifted_rates = {};
  flat_shifted_rates.resolution = resolution;
  flat_shifted_rates.sweep = [];

  for(var k=0; k<resolution+1;k++){

    var lr = lr_min + k/resolution*(lr_max-lr_min);
    //all_radius[k] = [];
    shifted_rates[k] = [];

    for(var l=0; l<resolution+1;l++){

      var beta = beta_min + l/resolution*(beta_max-beta_min);

      // for each eigenvalue
      var radius_eigs = 0;
      var radius_shifted_eigs = 0;

      for(var i=0;i<eigs.length;i++){
        var new_eigs = shift_eig(eigs[i], lr, beta);

        var transition_eig = math.subtract(1, math.multiply(lr, eigs[i]));

        // Get convergence circles
        var abs_eig = math.abs(transition_eig);
        var abs_shifted_eig = math.max(math.abs(new_eigs[0]), math.abs(new_eigs[1]));
        radius_eigs = math.max(radius_eigs, abs_eig);
        radius_shifted_eigs = math.max(radius_shifted_eigs, abs_shifted_eig);
      }

      shifted_rates[k][l] = radius_shifted_eigs;
      flat_shifted_rates.sweep.push({
        lr: lr,
        beta: beta,
        rate: radius_shifted_eigs
      });

    }
  }
  return flat_shifted_rates;
}


function refresh_touchpad(tp, lr, beta, shifted_rates) {
  // Plot heat map if shifted_rates is given
  if(shifted_rates) {
    var heatmapCell = tp.touchpad.selectAll('.heatmapCell')
    .data(shifted_rates.sweep);

    for(var i=0; i<shifted_rates.sweep.length;i++){
      shifted_rates.sweep[i];
    }
    tp.touchpad.selectAll('heatmapCell')
  }

  var lr_box = create_unique(tp.touchpad, 'text', 'lr', ['box'])
  .attr("x", 20)
  .attr("y", tp.pixels/2)
  .text('lr='+lr.toFixed(2));

  var beta_box = create_unique(tp.touchpad, 'text', 'beta', ['box'])
  .attr("x", tp.pixels/2)
  .attr("y", 40)
  .text('beta='+beta.toFixed(2));

  /////// bar for LR
  var lr_bar_origin = tp.yScale(0);
  var lr_bar_length = lr_bar_origin - Math.floor(tp.yScale(lr));
  var lr_bar_positive_length = Math.max(0, lr_bar_length);
  var lr_bar_negative_length = Math.max(0, -lr_bar_length);

  console.log('lr_bar_length', lr_bar_length);
  console.log('lr', lr);

  var lr_bar_positive = create_unique(tp.touchpad, 'rect', 'lr_bar_positive', ['bar_positive'])
  .attr("x", 0)
  .attr("y", lr_bar_origin-Math.floor(lr_bar_positive_length))
  .attr('width', tp.bar_thick)
  .attr('height', Math.floor(lr_bar_positive_length))
  .style('fill', 'green');

  var lr_bar_negative = create_unique(tp.touchpad, 'rect', 'lr_bar_negative', ['bar_negative'])
  .attr("x", 0)
  .attr("y", lr_bar_origin)
  .attr('width', tp.bar_thick)
  .attr('height', Math.floor(lr_bar_negative_length))
  .style('fill', 'red');


  /////// bar for Beta
  var beta_bar_origin = tp.xScale(0);
  var beta_bar_length = Math.floor(tp.xScale(beta)) - beta_bar_origin;
  var beta_bar_positive_length = Math.max(0, beta_bar_length);
  var beta_bar_negative_length = Math.max(0, -beta_bar_length);

  console.log('beta_bar_length', beta_bar_length);
  console.log('beta', beta);

  var beta_bar_positive = create_unique(tp.touchpad, 'rect', 'beta_bar_positive', ['bar_positive'])
  .attr("x", beta_bar_origin)
  .attr("y", 0)
  .attr('width', beta_bar_positive_length)
  .attr('height', tp.bar_thick)
  .style('fill', 'green');

  var beta_bar_negative = create_unique(tp.touchpad, 'rect', 'beta_bar_negative', ['bar_negative'])
  .attr("x", beta_bar_origin - beta_bar_negative_length)
  .attr("y", 0)
  .attr('width', beta_bar_negative_length)
  .attr('height', tp.bar_thick)
  .style('fill', 'red');
}
