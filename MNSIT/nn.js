let   tf  =  require("@tensorflow/tfjs") , 
      data  =  require("./data"); 

const model = tf.sequential({
  layers: [tf.layers.dense({units: 30, inputShape: [28*28] , activation:"relu"}),
          tf.layers.dense({units: 10 ,  activation :"softmax"}), ]
});


model.compile({
  optimizer: tf.train.sgd(0.1), 
  loss : "categoricalCrossentropy"
});


async function t(){
  const h = await model.fit(tf.tensor(data[0].xs), tf.tensor(data[0].ys) , {
    batchSize: 15,
    epochs: 30 , 
    callbacks :  {
      onBatchEnd :  (b , l )=> {
        console.log(l.loss);
        console.log(data[1][10][1])
        model.predict(tf.tensor([data[1][10][0]])).print()
      }
    }
}).catch(err => console.log(err));
}
t();


