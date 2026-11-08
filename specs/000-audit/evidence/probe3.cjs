const N=require(process.argv[2]);
const l=new N.LSTMLayer(2,2);const s=l.getWeights();const b=s.cellGate.W[0][0];l.predict([.1,.2]);l.backprop([[1,1]],.1);console.log('LSTM getWeights() snapshot mutó tras entrenar:',b!==s.cellGate.W[0][0]);
const c=new N.Conv2D(4,4,1,2,1,{padding:'same'});const o=c.forward(Array.from({length:4},()=>Array.from({length:4},()=>[1])));console.log("Conv2D same k=2 entrada 4x4 →",o.length,'x',o[0].length,'shape',c.outputShape());
const seq=[[.5,-.3],[.1,.7],[-.4,.2]],tg=[[.3],[.6],[.2]];
for(const lr of [.02,.002,.0005]){const n=new N.Seq2Seq(2,6,1);const a=n.trainStep(seq,tg,0);const hist=[];for(let e=0;e<300;e++)n.trainStep(seq,tg,lr);console.log('Seq2Seq lr',lr,a.toFixed(3),'→',n.trainStep(seq,tg,0).toExponential(2))}
