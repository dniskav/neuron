const N=require(process.argv[2]);const quick=process.argv[3]==='quick';
function seed(s){let a=s>>>0;Math.random=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const cos=(a,b)=>{let d=0,x=0,y=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];x+=a[i]*a[i];y+=b[i]*b[i]}return d/Math.sqrt(x*y||1e-300)};
const med=a=>[...a].sort((x,y)=>x-y)[a.length>>1];
const sq=(a,b)=>a.reduce((s,v,i)=>s+(v-b[i])**2,0)/2;
const G=(n)=>n.getWeights(),SW=(n,w)=>n.setWeights(w);
function gc(name,mk,loss,train,get=G,set=SW){const r=[];for(const s of [1,2,3,4,5]){seed(s);const n=mk();const w0=[...get(n)];const g=[];const e=1e-5;for(let i=0;i<w0.length;i++){const w=[...w0];w[i]+=e;set(n,w);const a=loss(n);w[i]-=2*e;set(n,w);const b=loss(n);g.push((a-b)/(2*e))}set(n,w0);loss(n);const lr=1e-6;train(n,lr);const d=get(n).map((v,i)=>(v-w0[i])/lr);r.push(cos(d,g.map(v=>-v)))}console.log(`${name.padEnd(40)} cos=${(r.reduce((a,b)=>a+b)/5).toFixed(4)}`)}
const ce=(n,tk,t,nc)=>{const o=n.predict(tk);let L=0;for(let i=0;i<tk.length;i++){const p=N.softmax(o.slice(i*nc,i*nc+nc));for(let c=0;c<nc;c++)if(t[i*nc+c]>0)L-=Math.log(p[c])}return L};
const tk=[0,1,2],tt=[1,0,0,1,1,0],qs=[[.1,.2],[.3,-.1],[.5,.4]];
const realStep=N.Adam.prototype.step,realEmb=N.EmbeddingMatrix.prototype.update;
const data=[[0,1,2,3],[3,2,1,0],[1,1,2,2],[0,3,0,3]],oh=t=>t.flatMap(v=>[0,1,2,3].map(c=>c===v?1:0));
function tf(lr,ep){const r=[];for(const s of [11,22,33,44,55]){seed(s);const n=new N.NetworkTransformer(4,{vocabSize:4,d_model:16,nHeads:2,d_ff:32,nBlocks:1,nClasses:4});const L=()=>data.reduce((a,t)=>a+ce(n,t,oh(t),4)/4,0)/4;const a=L();for(let e=0;e<ep;e++)data.forEach(t=>n.train(t,oh(t),lr));r.push([a,L()])}return `${med(r.map(x=>x[0])).toFixed(3)} → ${med(r.map(x=>x[1])).toFixed(3)}`}
function trl(lr,ep){const r=[];for(const s of [11,22,33,44,55]){seed(s);const n=new N.NetworkTransformerRL(3,2,{d_model:8,nHeads:2,d_ff:16,nBlocks:1,nActions:2});const L=()=>N.mse(n.predict(qs),[.3,.7]);const a=L();for(let e=0;e<ep;e++)n.train(qs,[.3,.7],lr);r.push([a,L()])}return `${med(r.map(x=>x[0])).toFixed(3)} → ${med(r.map(x=>x[1])).toFixed(3)}`}
console.log('== Transformer tal cual:  copy lr=.001',tf(.001,200),'| lr=.0002',tf(.0002,200),'| RL lr=.001',trl(.001,200));
N.Adam.prototype.step=function(w,g,lr){return realStep.call(this,w,-g,lr)};N.EmbeddingMatrix.prototype.update=function(i,g,lr){return realEmb.call(this,i,g.map(v=>-v),lr)};
console.log('== CONTROL signo invertido: copy lr=.001',tf(.001,200),'| lr=.0002',tf(.0002,200),'| RL lr=.001',trl(.001,200));
N.Adam.prototype.step=realStep;N.EmbeddingMatrix.prototype.update=realEmb;
if(quick)process.exit(0);
N.Adam.prototype.step=function(w,g,lr){return w+lr*g};
console.log('\n== Gradiente exacto (Adam→paso lineal solo para medir)');
gc('NetworkTransformer',()=>new N.NetworkTransformer(3,{vocabSize:4,d_model:4,nHeads:2,d_ff:6,nBlocks:1,nClasses:2}),n=>ce(n,tk,tt,2),(n,l)=>n.train(tk,tt,l));
gc('NetworkTransformerRL',()=>new N.NetworkTransformerRL(3,2,{d_model:4,nHeads:2,d_ff:6,nBlocks:1,nActions:2}),n=>sq(n.predict(qs),[.3,.7]),(n,l)=>n.train(qs,[.3,.7],l));
N.Adam.prototype.step=realStep;
const SGD=()=>new N.SGD(),X=[[.5,-.3,.8,.1],[-.2,.6,-.4,.3]],xs=[.5,-.3];
const sumsq=m=>m.reduce((s,r)=>s+r.reduce((a,b)=>a+b*b,0),0);
N.Adam.prototype.step=function(w,g,lr){return w+lr*g};
gc('AttentionHead (dOut=-dL)',()=>new N.AttentionHead(4,2,2),n=>sumsq(n.predict(X)),(n,l)=>{const o=n.predict(X);n.backward(o.map(r=>r.map(v=>-2*v)),l)});
N.Adam.prototype.step=realStep;
const i1=[.5,-.3,.8,.1,-.6];
gc('Conv1D (dOut=-dL)',()=>new N.Conv1D(5,3,2,1,'valid',SGD,1),n=>sumsq(n.forward(i1)),(n,l)=>{const o=n.forward(i1);n.backward(o.map(r=>r.map(v=>-2*v)),l)});
const img=[[[.1],[.5],[.2]],[[.7],[.3],[.9]],[[.4],[.6],[.8]]];const s3=o=>o.reduce((s,r)=>s+r.reduce((a,c)=>a+c.reduce((p,v)=>p+v*v,0),0),0);
gc('Conv2D (dOut=-dL)',()=>new N.Conv2D(3,3,1,2,2),n=>s3(n.forward(img)),(n,l)=>{const o=n.forward(img);n.backward(o.map(r=>r.map(c=>c.map(v=>-2*v))),l)});
const seq=[[.5,-.3],[.1,.7],[-.4,.2]],tg=[[.3],[.6],[.2]];
const rec=(n)=>{n.reset();return seq.map(x=>n.predict(x))};
gc('LSTMLayer BPTT (dh=-dL)',()=>new N.LSTMLayer(2,3,SGD),n=>sumsq(rec(n)),(n,l)=>{const h=rec(n);n.backprop(h.map(r=>r.map(v=>-2*v)),l)},n=>n.getWeightsFlat(),(n,w)=>n.setWeightsFlat(w));
gc('GRULayer BPTT (dh=-dL)',()=>new N.GRULayer(2,3,SGD),n=>sumsq(rec(n)),(n,l)=>{const h=rec(n);n.backprop(h.map(r=>r.map(v=>-2*v)),l)},n=>n.getWeightsFlat(),(n,w)=>n.setWeightsFlat(w));
const ax=[.2,.9,.4];const aeG=n=>n.encoder.getWeights(),aeS=(n,w)=>n.encoder.setWeights(w);
gc('Autoencoder sólo encoder sigmoid',()=>new N.Autoencoder(3,[4],2,[4]),n=>sq(n.reconstruct(ax),ax),(n,l)=>n.train(ax,l),aeG,aeS);
const flat=n=>{const o=[];for(const l of n.layers){const cf=[n=>n];for(const f of l.kernels)for(const k of f)for(const v of k)o.push(v);o.push(...l.biases)}for(const r of n._outputW)o.push(...r);o.push(...n._outputB);return o};
const setflat=(n,w)=>{let i=0;for(const l of n.layers){for(const f of l.kernels)for(const k of f)for(let c=0;c<k.length;c++)k[c]=w[i++];for(let b=0;b<l.biases.length;b++)l.biases[b]=w[i++]}for(const r of n._outputW)for(let c=0;c<r.length;c++)r[c]=w[i++];for(let b=0;b<n._outputB.length;b++)n._outputB[b]=w[i++]};
try{gc('TCN',()=>new N.TCN(2,3,2,2,1),n=>n.forward(seq).reduce((s,x,t)=>s+sq(x,tg[t]),0),(n,l)=>n.train(seq,tg,l),flat,setflat)}catch(e){console.log('TCN gc error:',e.message)}
console.log('\n== Control con factory de signo invertido (RNN, Seq2Seq, TCN)');
const flip=()=>({step:(w,g,lr)=>w-lr*g});
function lr3(mk,step){const r=[];for(const s of [11,22,33,44,55]){seed(s);const n=mk();const a=step(n,0);for(let e=0;e<500;e++)step(n,.02);r.push([a,step(n,0)])}return `${med(r.map(x=>x[0])).toFixed(3)} → ${(med(r.map(x=>x[1]))).toExponential(2)}`}
const rs=(n,l)=>n.backward(seq,tg,l),ss=(n,l)=>n.trainStep(seq,tg,l),ts=(n,l)=>n.train(seq,tg,l);
console.log('RNN      tal cual',lr3(()=>new N.RNN(2,6,1),rs),' | invertido',lr3(()=>new N.RNN(2,6,1,flip),rs));
console.log('TCN      tal cual',lr3(()=>new N.TCN(2,6,2,2,1),ts),' | invertido',lr3(()=>new N.TCN(2,6,2,2,1,flip),ts));
try{console.log('Seq2Seq  tal cual',lr3(()=>new N.Seq2Seq(2,6,1),ss),' | invertido',lr3(()=>new N.Seq2Seq(2,6,1,flip),ss))}catch(e){console.log('S2S',e.message)}
console.log('\n== Residual: XOR');
for(const [lbl,o] of [['residual sigmoid',{residual:true}],['sin residual',{}]]){const r=[];for(const s of [11,22,33,44,55]){seed(s);const n=new N.NetworkN([2,2,2,1],o);const XOR=[[0,0],[0,1],[1,0],[1,1]],T=[0,1,1,0];for(let e=0;e<8000;e++)XOR.forEach((x,i)=>n.train(x,[T[i]],.5));r.push(XOR.reduce((a,x,i)=>a+(n.predict(x)[0]-T[i])**2,0)/4)}console.log(lbl.padEnd(20),'mse mediana',med(r).toFixed(4),'max',Math.max(...r).toFixed(4))}
console.log('\n== VAE (config justa: tanh + salida lineal)');
const D=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];seed(1);const v=new N.VAE(4,[8],2,[8],{activations:[N.tanh,N.linear]});for(let e=0;e<5000;e++)D.forEach(x=>v.train(x,.02));console.log('mu',JSON.stringify(D.map(x=>v.encode(x).mu.map(m=>+m.toFixed(3)))),'recon',D.reduce((a,x)=>a+N.mse(v.decode(v.encode(x).mu),x),0)/4,'(predecir la media da 0.1875)');
console.log('\n== Sondas');
const T=(n,f)=>{try{console.log(n,f())}catch(e){console.log(n,'→ THROW:',e.message)}};
T('NetworkN.setWeights corto (2 capas):',()=>{seed(9);const n=new N.NetworkN([2,2,1]);n.setWeights([1,2,3]);return n.predict([1,1])});
T('NetworkN.setWeights corto (1 capa):',()=>{const n=new N.NetworkN([2,1]);n.setWeights([1]);return n.predict([1,1])});
T('ModelSaver pesos de otro modelo sin error (largos):',()=>{const a=new N.NetworkN([2,4,1]),b=new N.NetworkN([2,2,1]);N.ModelSaver.fromJSON(b,N.ModelSaver.toJSON(a));return 'aceptado, predict='+b.predict([1,1])});
T('LSTM getWeights referencia viva:',()=>{const l=new N.LSTMLayer(2,2);const s=l.getWeights();const b=s.forgetGate.W[0][0];l.predict([.1,.2]);l.backprop([[1,1]],.1);return b!==s.forgetGate.W[0][0]});
T('Network 2 salidas, 2ª entrenada:',()=>{const n=new N.Network(2,3,2);const o=[...n.outputLayer.neurons[1].weights];for(let i=0;i<100;i++)n.train([1,0],1,.5);return o.some((v,i)=>v!==n.outputLayer.neurons[1].weights[i])});
T('Tape 20k nodos:',()=>{let v=new N.Value(1);for(let i=0;i<20000;i++)v=v.add(1);v.backward();return 'ok'});
T("Conv2D same kernel par 4x4 →",()=>new N.Conv2D(4,4,1,1,2,{padding:'same'}).outputShape());
T("Conv1D same kernel par len4 →",()=>{const c=new N.Conv1D(4,2,1,1,'same');return [c.getOutputLength(),c.forward([1,2,3,4])[0].length]});
T('Trainer earlyStopping sin validación:',()=>{const w=console.warn;let c=0;console.warn=()=>c++;const t=new N.Trainer(new N.NetworkN([2,2,1]),{epochs:3,earlyStopping:{patience:1,minDelta:0}});t.train({inputs:[[0,0]],targets:[[0]]});console.warn=w;return `avisos=${c} stop=${t.getStopReason()}`});
console.log('\n== Clásicos');
seed(3);const blobs=[];const yb=[];for(let i=0;i<60;i++){const c=i%2;blobs.push([c*5+Math.random(),c*5+Math.random()]);yb.push(c)}
const acc=(p)=>blobs.reduce((a,x,i)=>a+(p(x)===yb[i]?1:0),0)/60;
T('DecisionTree XOR acc',()=>{const t=new N.DecisionTree();const X=[[0,0],[0,1],[1,0],[1,1]],y=[0,1,1,0];t.fit(X,y);return X.filter((x,i)=>t.predict(x)===y[i]).length/4});
T('Perceptron blobs acc',()=>{const p=new N.Perceptron(2);p.train(blobs,yb,.1,50);return acc(x=>p.predict(x))});
T('LogReg blobs acc',()=>{const c=new N.LogisticRegression(2);c.train(blobs,yb,.05,300);return acc(x=>c.classify(x))});
T('Softmax blobs acc',()=>{const c=new N.SoftmaxRegression(2,2);c.train(blobs,yb,.05,300);return acc(x=>c.predict(x))});
T('NaiveBayes acc',()=>{const c=new N.GaussianNaiveBayes();c.fit(blobs,yb);return acc(x=>c.predict(x))});
T('KMeans pureza',()=>{const k=new N.KMeans(2);k.fit(blobs);const l=k.predictBatch(blobs);const m=blobs.filter((_,i)=>l[i]===yb[i]).length/60;return Math.max(m,1-m)});
T('PCA varianza 1ª comp (esperado≈.98)',()=>{const p=new N.PCA(2);p.fit(blobs);return p.explainedVarianceRatio()});
T('LinearRegression y=2x+1',()=>{const l=new N.LinearRegression();const X=[[1],[2],[3],[4]],y=[3,5,7,9];const f=l.fit||l.train;f.call(l,X,y);return l.predict([10])});
T('Hopfield recall',()=>{const h=new N.HopfieldNetwork(16);const p=Array.from({length:16},(_,i)=>i%3?1:-1);h.store(p);const q=[...p];q[0]*=-1;q[5]*=-1;return h.recall(q).every((v,i)=>v===p[i])});
T('Word2Vec king~queen vs man',()=>{const w=new N.Word2Vec(16,{model:'skipgram',windowSize:2});const c=[];for(let i=0;i<30;i++){c.push(["the","king","rules","the","kingdom"]);c.push(["the","queen","rules","the","land"]);c.push(["man","and","woman","are","human"])}w.buildVocab(c);w.train(c,.05,20);return [w.similarity('king','queen').toFixed(3),w.similarity('king','human').toFixed(3)]});
T('TSNE separa blobs',()=>{const t=new N.TSNE({perplexity:5,nIter:250});const f=t.fitTransform||t.fit;const o=f.call(t,blobs.slice(0,30));const m=c=>{const a=o.filter((_,i)=>yb[i]===c);return a.reduce((s,p)=>[s[0]+p[0]/a.length,s[1]+p[1]/a.length],[0,0])};const[a,b]=[m(0),m(1)];return Math.hypot(a[0]-b[0],a[1]-b[1]).toFixed(2)});
T('auc perfecto (esperado 1)',()=>N.auc(N.rocCurve([.9,.8,.2,.1],[1,1,0,0])));
T('Rendimiento NetworkN [100,256,10] Adam ms/muestra',()=>{seed(5);const b=new N.NetworkN([100,256,10],{optimizer:()=>new N.Adam()});const i=Array(100).fill(.5),t=Array(10).fill(.5);const t0=performance.now();for(let k=0;k<200;k++)b.train(i,t,.001);return ((performance.now()-t0)/200).toFixed(2)});
T('Rendimiento Transformer default 81 tokens ms/muestra',()=>{seed(5);const n=new N.NetworkTransformer(81,{});const t0=performance.now();n.train(Array(81).fill(1),Array(729).fill(0),.001);return (performance.now()-t0).toFixed(0)});
