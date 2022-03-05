
let   tf  =  require("@tensorflow/tfjs") , 
     data  =  require("./data");



let model =  tf.sequential()


model.add(tf.layers.conv2d({
    inputShape: [28 ,  28 ,  1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer  
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  //Repeat another conv2d + maxPooling stack. 
  
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  
  model.add(tf.layers.flatten());

  


  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax'
  }));

  

  model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });


  async function t(){
    const h = await model.fit(tf.tensor(data[0].xs,[data[0].xs.length , 28, 28 , 1]) , tf.tensor(data[0].ys) , {
      batchSize: 15,
      epochs: 3 , 
      shuffle :  true ,  
      callbacks :  {
        onEpochEnd :  (b , l )=> {
          console.log(l.loss);
          console.log(data[1][0][1])
          model.predict(tf.tensor(data[1][0][0],[1, 28, 28 , 1])).print()
        } , 
        onTrainEnd :  ()  => {
          console.log("model  train  finished ! "); 
          //model.save("./")
        }
      }
  }).catch(err => console.error(err));
  }
  t();