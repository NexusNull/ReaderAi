const convnetjs = require("convnetjs");
const Jimp = require('jimp');
const alph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split("");
const fs = require("fs");


function createImageVol(char){
    return new Promise(function(resolve, reject){
        new Jimp(20, 20, 0xffffffff, function(err, image) {
            Jimp.loadFont(Jimp.FONT_SANS_16_BLACK).then(function(font) {
                image.print(font, 2+Math.random()*8,  Math.random()*8-4, char);
                //image.write("./images/"+char+".png");
                let v = new convnetjs.Vol(20, 20, 1, 0.0);
                let data = [...image.bitmap.data];
                v.w =[];
                for(let i=0;i<data.length/4;i++){
                    v.w[i] = 1-(data[i*4]/255);
                }
                resolve(v);
            });
        });
    })
}

function chooseRandomLetter(){
    return alph[Math.floor(Math.random()*alph.length)];
}

function randomText(length){
    let string = "";
    for(let i=0;i<length; i++)
        string+=chooseRandomLetter();
    return string;
}

function saveNetwork(network,name){
    return new Promise(function(resolve,reject){
        fs.writeFileSync((name)?name:"network.json", JSON.stringify(network))
        resolve();
    })
}
function loadNetwork(name){
    return new Promise(function(resolve,reject){
        let data = fs.readFileSync((name)?name:"network.json");
        let network = new convnetjs.Net();
        network.fromJSON(JSON.parse(data));
        resolve(network);
    })
}

function createNetwork(){
    let layer_defs = [];
    layer_defs.push({type:'input', out_sx:20, out_sy:20, out_depth:1});
    layer_defs.push({type: 'conv', sx: 5, filters: 8, stride: 1, pad: 0, activation: 'relu'});
    layer_defs.push({type: 'pool', sx: 2, stride: 2});
    layer_defs.push({type: 'conv', sx: 5, filters: 16, stride: 1, pad: 0, activation: 'relu'});
    layer_defs.push({type: 'pool', sx: 3, stride: 3});
    layer_defs.push({type:'softmax', num_classes:alph.length});
    let network = new convnetjs.Net();
    network.makeLayers(layer_defs);
    return network;
}


async function main(){
    var network;
    let loadFromDisk = true;
    if(loadFromDisk){
        network = await loadNetwork();
    } else {
        network = createNetwork();
    }

    const trainingSet = [];
    const testSet = [];
    console.log("Creating Training Data")
        
    for(let i=0;i<1000;i++){
        let char = chooseRandomLetter();
        trainingSet[i] = {img: await createImageVol(char), label:char};
    }

    console.log("Creating Test Data")
    for(let i=0;i<1000;i++){
        let char = chooseRandomLetter();
        testSet[i] = {img: await createImageVol(char), label:char};
    }

    fs.writeFileSync("a.json",JSON.stringify(testSet[5]));
    var trainer = new convnetjs.SGDTrainer(network, {method:"adadelta", l2_decay:0.001, batch_size:40});
    while(true){
        //Training
        for(let j=0;j<1;j++){
            for(let i=0;i<trainingSet.length;i++){
                trainer.train(trainingSet[i].img, alph.indexOf(trainingSet[i].label));
            }
            process.stdout.write(".")
        }
        //Confusion matrix
        var conf = {};
        for(var i=0;i<alph.length;i++){
            conf[alph[i]] = {};
            for(let j=0;j<alph.length;j++){
                conf[alph[i]][alph[j]] = 0;
            }
        }

        //Testing
        var fail = 0;
        var success = 0;
        for(let i=0;i<testSet.length;i++){
            network.forward(testSet[i].img);
            var result = network.getPrediction();
            conf[alph[result]][testSet[i].label]++;

            if (result === alph.indexOf(testSet[i].label)) {
                success++;
            } else {
                fail++;
            }
        }
        console.log("\nFail: "+fail);
        console.log("Success: "+success);
        for(let i=0;i<alph.length;i++){
            process.stdout.write("\t"+alph[i]+"");
        }
        process.stdout.write("\n")
        for(let i=0;i<alph.length;i++){
            process.stdout.write(alph[i]);
            for(let j=0;j<alph.length;j++){
                process.stdout.write("\t"+conf[alph[i]][alph[j]]+"");
            }
            process.stdout.write("\n")
        }
        await saveNetwork(network);
    }

}

main();