/**
 * Created by Nexus on 12.10.2017.
 */
var gm = require("gm");
var convnetJs = require("convnetjs");
var fs = require('fs'),
    PNG = require('pngjs').PNG;
var networks = [];
var request = require("request-promise-native");

async function formatImage(i) {
    return new Promise(function (resolve, reject) {
        gm("download/captcha" + i + ".gif").setFormat("png").write("download/captcha" + i + ".png", function (error) {
            if (!error)
                resolve();
            else
                reject(error);
        });
    })
}

async function createNetworks(size) {
    for (var i = 0; i < 6; i++) {
        var layer_defs = [];
        layer_defs.push({type: 'input', out_sx: size.width, out_sy: size.height, out_depth: 1});
        layer_defs.push({type: 'conv', sx: 5, filters: 8, stride: 1, pad: 0, activation: 'relu'});
        layer_defs.push({type: 'pool', sx: 2, stride: 2});
        layer_defs.push({type: 'conv', sx: 5, filters: 16, stride: 1, pad: 0, activation: 'relu'});
        layer_defs.push({type: 'pool', sx: 3, stride: 3});
        layer_defs.push({type: 'softmax', num_classes: 10});

        networks[i] = new convnetJs.Net();
        networks[i].makeLayers(layer_defs);
    }
}

async function saveNetworks() {
    for (var i = 0; i < networks.length; i++) {
        fs.writeFileSync("networks/network" + i + ".json", JSON.stringify(networks[i].toJSON()));
    }
}

async function loadNetworks() {
    var i = 0;
    while (fs.existsSync("networks/network" + i + ".json")) {
        var data = fs.readFileSync("networks/network" + i + ".json");
        networks[i] = new convnetJs.Net();
        networks[i].fromJSON(JSON.parse(data));
        i++;
    }
}

async function downloadImage(number, filename) {
    return new Promise(function (resolve) {
        request("http://localhost/assets/image.php?n=" + number, {}).pipe(fs.createWriteStream('train/train.png')).on("close", resolve);
    })
}

async function filter(){
    var img = [];
    await Promise.all([
        new Promise(function (resolve) {
            fs.createReadStream('download/captcha1.png')
                .pipe(new PNG({
                    filterType: 4
                }))
                .on('parsed', function () {
                    for (var i = 0; i < this.width * this.height; i++) {
                        if (this.data[i * 4] < 100)
                            img[i] = this.data[i * 4];
                        else
                            img[i] = 255;
                    }
                    resolve();
                });
        }),
        new Promise(function (resolve) {
            fs.createReadStream('download/captcha2.png')
                .pipe(new PNG({
                    filterType: 4
                }))
                .on('parsed', function () {
                    for (var i = 0; i < this.width * this.height; i++) {
                        if (this.data[i * 4] < 100);
                        else
                            img[i] = 255;
                    }
                    resolve();
                });
        }),
        new Promise(function (resolve) {
            fs.createReadStream('download/captcha3.png')
                .pipe(new PNG({
                    filterType: 4
                }))
                .on('parsed', function () {
                    for (var i = 0; i < this.width * this.height; i++) {
                        if (this.data[i * 4] < 100);
                        else
                            img[i] = 255;
                    }
                    resolve();
                });
        }),
    ]);
    for (var i = 0; i < img.length; i++) {
        img[i] = img[i] / 255;
    }
    return img;
}

async function parse(filename) {
    return new Promise(function (resolve) {
        fs.createReadStream(filename)
            .pipe(new PNG({
                filterType: 4
            }))
            .on('parsed', function () {
                var img = [];
                for (var i = 0; i < this.width * this.height; i++) {
                    img[i] = this.data[i * 4] / 255;
                }
                resolve(img);
            });
    });
}

async function trainNetwork() {
    await loadNetworks();
    //await createNetworks({width: 66, height: 20})
    var trainers = [];
    var trainsData = [];
    var testData = [];
    for (var i = 0; i < networks.length; i++) {
        trainers[i] = new convnetJs.SGDTrainer(networks[i], {method: 'adadelta', l2_decay: 0.001, batch_size: 30});
    }

    for (let k = 0; k < 1000; k++) {
        let number = parseInt(Math.random() * (1000000 - 99999) + 99999);
        await downloadImage(number, 'train/train.png');
        let img = await parse('train/train.png');
        let vol = new convnetJs.Vol(66, 20, 1, 0.0);
        vol.w = img;

        trainsData[k] = {
            img: vol, label: (number + "").split("").map(function (a) {
                return parseInt(a);
            })
        };
    }

    for (let k = 0; k < 100; k++) {
        let number = parseInt(Math.random() * (1000000 - 99999) + 99999);
        await downloadImage(number, 'train/train.png');
        let img = await parse('train/train.png');
        let vol = new convnetJs.Vol(66, 20, 1, 0.0);
        vol.w = img;

        testData[k] = {
            img: vol, label: (number + "").split("").map(function (a) {
                return parseInt(a);
            })
        };
    }

    for (let i = 0; i < 10; i++) {
        shuffle(trainsData);
        shuffle(testData);
        for (let t = 0; t < trainsData.length; t++) {
            for (let i = 0; i < networks.length; i++) {
                var result = trainers[i].train(trainsData[t].img, trainsData[t].label[i]);
            }
        }
        console.log("------------------------------------------------------------------------------------------------");
        var x = 0;
        var _ = 0
        for (var t = 0; t < testData.length; t++) {
            process.stdout.write(testData[t].label.join(""));
            for (let i = 0; i < networks.length; i++) {
                networks[i].forward(testData[t].img);
                var result = networks[i].getPrediction();
                if (result === testData[t].label[i]) {
                    process.stdout.write("_");
                    _++;
                } else {
                    process.stdout.write("x");
                    x++;
                }
            }
            process.stdout.write("\n")
        }
        console.log("x:"+x+" _:"+_);
    }

    await saveNetworks();
}

module.exports.train = trainNetwork;
module.exports.learn = function(guess, result){
    fs.createReadStream('output/'+guess+'.png').pipe(fs.createWriteStream('solved/'+result+'.png'));
};

module.exports.guess = async function () {
    await formatImage(1);
    await formatImage(2);
    await formatImage(3);
    var img = await filter();

    var output = new PNG({width:66,height:20})
    for(var i=0;i<img.length;i++){
        output.data[i*4] = img[i]*255;
        output.data[i*4+1] = img[i]*255;
        output.data[i*4+2] = img[i]*255;
        output.data[i*4+3] = 255;
    }

    await loadNetworks();

    var vol = new convnetJs.Vol(66, 20, 1, 0.0);
    vol.w = img;
    var result = "";

    for (var i = 0; i < networks.length; i++) {
        var net = networks[i];
        net.forward(vol);
        result += net.getPrediction();
    }
    output.pack().pipe(fs.createWriteStream('output/'+result+'.png'));

    return result;
};

var shuffle = function (a) {
    var j, x, i;
    for (i = a.length; i; i--) {
        j = Math.floor(Math.random() * i);
        x = a[i - 1];
        a[i - 1] = a[j];
        a[j] = x;
    }
};