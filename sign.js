const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const predictionElement = document.getElementById('prediction');

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({
  poolSize: 2,
  strides: 2,
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({
  poolSize: 2,
  strides: 2,
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
  units: 128,
  activation: 'relu',
}));
model.add(tf.layers.dense({
  units: 10,
  activation: 'softmax',
}));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

// Start the video stream and get the video dimensions
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
  video.onloadedmetadata = () => {
    video.play();
    
    // Set the canvas dimensions to match the video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Create a canvas context for drawing
    const ctx = canvas.getContext('2d');
    
    // Start the prediction loop
    setInterval(() => {
      // Draw the current video frame on the canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get the canvas image data and convert it to a Tensor
      // const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      // const tensor = tf.browser.fromPixels(imageData).reshape([1, 28, 28, 1]).toFloat().div(255);
      const imageData = canvas.toDataURL();
      const input = tf.browser.fromPixels(canvas, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(255.0)
        .reshape([1, 28, 28, 1]);

      
      // Make the prediction and get the predicted digit
      const prediction = model.predict(input).argMax(1).dataSync()[0];
      
      // Display the predicted digit
      predictionElement.textContent = prediction;
    }, 1000 / 30);
  };
}).catch(error => {
  console.error(error);
});
