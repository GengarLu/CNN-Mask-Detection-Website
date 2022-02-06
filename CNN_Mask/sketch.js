let video;
let videoSize = 32;
let ready = false;

let pixelBrain;
let label = '';

function setup() {//設定影片大小
  createCanvas(400, 400);
  video = createCapture(VIDEO, videoReady);
  video.size(videoSize, videoSize); //像素為10*10
  video.hide();

  //kernel Size:
  //我們如何從點轉成面呢? 很簡單，就是以圖像的
  //每一點為中心，取周遭 N x N 格的點構成一個面
  //(N稱為 Kernel Size，N x N 的矩陣權重稱為『卷積核』)

  //kernel Initializer:
  //一開始我們會選擇一個點開始，此點即稱為
  //『權重的初始值』(kernel_initializer)，
  //初始值的選擇可能會影響優化的結果
  let customLayers = [//設定卷積層
    {
      type: 'conv2d',
      filters: 4,
      kernelSize: 5,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    },
    {
      type: 'maxPooling2d',
      poolSize: [2, 2],
      strides: [2, 2],
    },
        {
      type: 'conv2d',
      filters: 8,
      kernelSize: 5,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    },
    {
      type: 'maxPooling2d',
      poolSize: [2, 2],
      strides: [2, 2],
    },
    {
      type: 'flatten',
    },
    {
      type: 'dense',
      kernelInitializer: 'varianceScaling',
      activation: 'softmax',
    },
  ];

  let options = {//設置神經網路
    inputs: [32, 32, 4], //r,g,b,透明度
    task: 'imageClassification',
    layers: customLayers,
    debug: true //在模型訓練時看到調試
  };
  //而ml5的神經網路會根據你輸入和輸出的總數來指定數量，所以不用自己調
  
  pixelBrain = ml5.neuralNetwork(options); //調用ml5的神經網路
  pixelBrain.loadData('model/test.json', loaded);//載入訓練集
  const modelInfo = {//已訓練模型的檔案位置
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  pixelBrain.load(modelInfo, modelLoaded);//載入已訓練模型
}

// Video is ready!
function videoReady() {
  ready = true;
}

function loaded() {//載入訓練集
  pixelBrain.normalizeData();
  // pixelBrain.train({ epochs: 50 }, finishedTraining);
}

function modelLoaded() {
  classifyVideo();//當已訓練模型載入時，只會把神經網路架構、輸入節點數、輸出節點數、one hot coding.....載入，所以需要串接classifyVideo這個函式以不斷循環讀取label並顯示
  console.log('Model Loaded!');
}

function modelSaved() {//儲存已訓練模型後會顯示Model Saved!
  console.log('Model Saved!')
}
function addExample(label) {//為訓練集加上用來分類的標籤
  let inputImage = { image: video };
  let target = { label }; //在這屬性名稱恰好=值的變量名稱，所以在javascript可以省略label:
  console.log('Adding example: ' + label);
  pixelBrain.addData(inputImage, target); //將inputs和target放入資料中
}

function classifyVideo() {//分類
  let inputImage = { image: video };
  pixelBrain.classify(inputImage, gotResults);
}

function finishedTraining() {//結束訓練
  console.log('training complete');
  classifyVideo();
}

function gotResults(error, results) {//得知結果
  if (error) {
    return;
  }
  console.log(results);
  label = results[0].label;
  classifyVideo(); //要記得再次對視頻分類
}

function keyPressed() {//每個按鍵對應的動作
  if (key == 't') {//訓練
    pixelBrain.normalizeData(); //因像素值在0~255之間，所以要標準化到0~1之間
    pixelBrain.train({ epochs: 50 }, finishedTraining);
  } else if (key == 's') {//存測資
    pixelBrain.saveData();
  } else if (key == 'm') {//存有戴口罩
    addExample("Nice Mask");
  } else if (key == 'n') {//存沒戴口罩
    addExample("Please wear a mask!");
  } else if (key == 'e') {//存無人時
    addExample("Empty!");
  } else if (key == 'a') {//存已訓練模型
    pixelBrain.save(modelSaved);
  }
}

function draw() { //在畫布中為每個像素繪製一個矩形
  background(0);
  if (ready) {
    image(video, 0, 0, width, height); //0,0是左上角座標
  }
  if (label == "Nice Mask") {
    textSize(32);
    textAlign(CENTER, CENTER);
    fill(0, 255, 0);
    text(label, width / 2, height / 2);
  }
  else if (label == "Please wear a mask!") {
    textSize(32);
    textAlign(CENTER, CENTER);
    fill(255, 0, 0);
    text(label, width / 2, height / 2);
  }
  else if (label == "Empty!") {
    textSize(32);
    textAlign(CENTER, CENTER);
    fill(0, 0, 255);
    text(label, width / 2, height / 2);
  }
}