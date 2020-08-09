const imageUpload = document.getElementById('imageUpload')
let container;
let faceMatcher;
let canvas1=document.getElementById("can1");
$("#but1").on("click touchend",function (){
  $( ".train" ).toggle();
});
///////////////////////////////////////////////////////////////////

window.addEventListener("load", init);
function init(){
  container = document.createElement('div')
  container.style.position = 'relative'
  container.id="output";
  document.body.append(container);
const captureVideoButton = document.querySelector('#screenshot .capture-button');
const screenshotButton = document.querySelector('#screenshot-button');

const video = document.querySelector('#screenshot .webcam');
const canvas1 = document.querySelector('#screenshot #can2');
let img = new Image();


captureVideoButton.addEventListener('click', async (e) => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        minAspectRatio: 1.333,
        minFrameRate: 60,
        width: 200,
        heigth: 300
      }
    })  
    document.querySelector('video').srcObject = stream
  } catch(e) {
    console.error(e)
  }
});

screenshotButton.onclick = video.onclick = function() {
  canvas1.width = video.videoWidth;
  canvas1.height = video.videoHeight;
  canvas1.getContext('2d').drawImage(video, 0, 0);
  // Other browsers will fall back to image/png
  img.src = canvas1.toDataURL('image/webp');
  // img.width="300";
  // container.append(img);
  snapshot(img);
  
};

$("#scan").on("click touchend",function(){
img.src='test_images/4.jpeg'
  snapshot(img);
});
function handleSuccess(stream) {
  screenshotButton.disabled = false;
  video.srcObject = stream;
}

}

///////////////////////////////////////////////////////////////


Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('models')
]).then(start)

async function start() {
  console.log("loading Files");
  $( "#infoText" ).text("Training Started... please wait...");
  const labeledFaceDescriptors = await loadLabeledImages()
  faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
 
  $( "#infoText" ).text("Training complete");

}

imageUpload.addEventListener('change', async () => {
  $( "#can3" ).remove();
  $( "#img3" ).remove();
  let image
  image = await faceapi.bufferToImage(imageUpload.files[0])
  image.id = "img3";
  image.width="300";
  container.append(image)
  canvas = faceapi.createCanvasFromMedia(image)
  canvas.id="can3";
  $("#name").text("Detected:");
  container.append(canvas)
  const displaySize = { width: image.width, height: image.height }
  faceapi.matchDimensions(canvas, displaySize)
  const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
  const resizedDetections = faceapi.resizeResults(detections, displaySize)
  const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box
    const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
    drawBox.draw(canvas)
    $("#name").text("Detected:"+result.toString());
   
  })
})

async function snapshot(img) {
  const container = document.querySelector('#output')
  $( "#img3" ).remove();
  $( "#can3" ).remove();
  img.id="img3";
  // img.width="300";
  img.height="300";
  container.append(img)

//  var ctx = canvas1.getContext("2d");
//  ctx.clearRect(0, 0, canvas1.width, canvas1.height);
//  ctx.drawImage(img, 0, 0,canvas1.width,canvas1.height);
 
$("#name").text("Detected:");
 const displaySize = { width: 300, height: 300 }
  // faceapi.matchDimensions(canvas1, displaySize)
  const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors()
  const resizedDetections = faceapi.resizeResults(detections, displaySize)
  const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box
    const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
$("#name").text("Detected:"+result.toString());
// ctx.beginPath();
// ctx.rect(parseInt(box._x), parseInt(box._y), parseInt(box._width), parseInt(box._height));
// ctx.stroke();
    
   
  })
};

//Descriptor detection
function loadLabeledImages() {
  console.log("Started Training");
  const labels = ['Nikhil', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`labeled_images/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }
      console.log("Training Complete");
      
      let dat= new faceapi.LabeledFaceDescriptors(label, descriptions)
      return dat;
    })
  )
}
