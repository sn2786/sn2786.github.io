const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function getTextColor(red, green, blue) {
    var brightness = ((red * 299) + (green * 587) + (blue * 114)) / 1000;
    if(brightness < 125) { //background is dark
        return 'white';
    } else {
        return 'black';
    }
}

function doPredict(predict) {
  var textField = document.getElementById('text-entry');
  var btn = document.getElementById("btn");
  var view = document.getElementById("view");
  var msg = document.getElementById("message");
  console.log(textField.value);
  const result = predict(textField.value);
  score_string = "Class scores: ";
  for (var x in result.score) {
    score_string += x + " ->  " + result.score[x].toFixed(3) + ", "
  }
  console.log(score_string);
  const red = parseInt(result.score[0] * 255);
  const green = parseInt(result.score[1] * 255);
  const blue = parseInt(result.score[2] * 255);
  console.log("rgb: " + red+", " + green+", " + blue );
  var displayColor = function() {
      msg.style.display = "none";
      view.innerHTML =
          '<div class="' + getTextColor(red, green, blue) + '">' +
              '<div class="inner">' +
                  '<span>' + 'RGB(' + red + ', ' + green + ', ' + blue + ')' + '</span>' +
              '</div>' +
          '</div>' + view.innerHTML;
      view.firstChild.style.backgroundColor = 'RGB(' + red + ', ' + green + ', ' + blue + ')';
      textField.value = "";
    };
  btn.onclick = displayColor;
  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function prepUI(predict) {
  setPredictFunction(predict);
  settextField('tensorflow orange', predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    console.log(model.summary());
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.log(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {
  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split("");
    // Look up word indices.
    console.log(inputText);
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
   for (let i = 0; i < this.maxLen - inputText.length; ++i) {
      inputBuffer.set(0, 0, i);
    }
    for (let i = this.maxLen - inputText.length; i < this.maxLen; ++i) {
      const word = inputText[i - (this.maxLen - inputText.length)];
      inputBuffer.set(this.wordIndex[word], 0, i);
      console.log(word, this.wordIndex[word], inputBuffer);
    }
    console.log("inputBuffer: " + inputBuffer);
    const input = inputBuffer.toTensor();
   // input = tf.where(tf.math.is_nan(input), tf.zeros_like(input), input);
    console.log("input tensor: " +input + ", input shape:" + input.shape);
    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    console.log(score);
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');
    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();
