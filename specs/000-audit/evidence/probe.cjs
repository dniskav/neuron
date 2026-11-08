const N=require(process.argv[2]+'/dist/index.js');
function seed(s){let a=s>>>0;Math.random=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const SKIP=/opt|traj|cache|^h$|^c$|acts|input$|last|layerInputs|layerOutputs|padded|outputs$|structure|mask|dropout|_h$|^act|hist/i;
function collect(o,seen=new Set(),out=[]){if(!o||typeof o!=='object'||seen.has(o))return out;seen.add(o);
 if(Array.isArray(o)){if(o.length&&typeof o[0]==='number'){o.forEach((_,i)=>out.push([o,i]));return out} o.forEach(x=>collect(x,seen,out));return out}
 for(const k of Object.keys(o).sort()){if(SKIP.test(k))continue;const v=o[k];if(typeof v==='function')continue; if(typeof v==='number'&&/^bias$/.test(k)){out.push([o,k]);continue} collect(v,seen,out)}return out}
const read=n=>collect(n).map(([a,i])=>a[i]);
function write(n,w){collect(n).forEach(([a,i],k)=>{a[i]=w[k]})}
function cos(a,b){let d=0,x=0,y=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];x+=a[i]*a[i];y+=b[i]*b[i]}return d/Math.sqrt(x*y||1e-300)}
function gradcheck(name,mk,loss,train){const rows=[];for(const s of [1,2,3,4,5]){seed(s);const n=mk();const w0=read(n);const g=[];const e=1e-5;
  for(let i=0;i<w0.length;i++){const w=[...w0];w[i]+=e;write(n,w);const a=loss(n);w[i]-=2*e;write(n,w);const b=loss(n);g.push((a-b)/(2*e))}
  write(n,w0);loss(n);const lr=1e-6;train(n,lr);const w1=read(n);const d=w1.map((v,i)=>(v-w0[i])/lr);rows.push(cos(d,g.map(v=>-v)))}
 const m=rows.reduce((a,b)=>a+b)/rows.length;console.log(`${name.padEnd(36)} cos(Δw,-∇L)=${m.toFixed(4)}  min=${Math.min(...rows).toFixed(4)}`)}
const realAdam=N.Adam.prototype.step;
const x3=[.5,-.3,.8],sq=(a,b)=>a.reduce((s,v,i)=>s+(v-b[i])**2,0)/2;
console.log('== A. Gradiente real vs numérico (1=descenso correcto, -1=ascenso)');
gradcheck('Network sigmoid',()=>new N.Network(3,4,1),n=>sq(n.predict(x3),[.7]),(n,l)=>n.train(x3,.7,l));
gradcheck('NetworkN sigmoid',()=>new N.NetworkN([3,4,3,2]),n=>sq(n.predict(x3),[.7,.2]),(n,l)=>n.train(x3,[.7,.2],l));
gradcheck('NetworkN tanh/elu/linear',()=>new N.NetworkN([3,4,3,2],{activations:[N.tanh,N.elu,N.linear]}),n=>sq(n.predict(x3),[.7,.2]),(n,l)=>n.train(x3,[.7,.2],l));
gradcheck('NetworkN residual sigmoid',()=>new N.NetworkN([3,3,3,1],{residual:true}),n=>sq(n.predict(x3),[.7]),(n,l)=>n.train(x3,[.7],l));
gradcheck('NetworkN residual tanh+linear',()=>new N.NetworkN([3,3,3,1],{residual:true,activations:[N.tanh,N.tanh,N.linear]}),n=>sq(n.predict(x3),[.7]),(n,l)=>n.train(x3,[.7],l));
gradcheck('NetworkN salida residual',()=>new N.NetworkN([3,3,3],{residual:true}),n=>sq(n.predict(x3),[.7,.1,.3]),(n,l)=>n.train(x3,[.7,.1,.3],l));
const ax=[.2,.9,.4];
gradcheck('Autoencoder sigmoid',()=>new N.Autoencoder(3,[4],2,[4]),n=>sq(n.reconstruct(ax),ax),(n,l)=>n.train(ax,l));
gradcheck('Autoencoder tanh latente',()=>new N.Autoencoder(3,[4],2,[4],{activations:[N.tanh,N.tanh,N.tanh,N.sigmoid]}),n=>sq(n.reconstruct(ax),ax),(n,l)=>n.train(ax,l));
const seq=[[.5,-.3],[.1,.7],[-.4,.2]],tg=[[.3],[.6],[.2]];
const lstmLoss=n=>{n.resetState();return seq.reduce((s,x,t)=>s+sq(n.predict(x),tg[t]),0)};
gradcheck('NetworkLSTM',()=>new N.NetworkLSTM(2,3,[2,1]),lstmLoss,(n,l)=>{lstmLoss(n);n.train(tg,l)});
const rnn=n=>{n.reset();const o=n.forward(seq).outputs;return o.reduce((s,x,t)=>s+sq(x,tg[t]),0)};
gradcheck('RNN',()=>new N.RNN(2,3,1),rnn,(n,l)=>n.backward(seq,tg,l));
const tcnL=n=>n.forward(seq).reduce((s,x,t)=>s+sq(x,tg[t]),0);
try{gradcheck('TCN',()=>new N.TCN(2,3,1,2,2),tcnL,(n,l)=>n.train(seq,tg,l))}catch(e){console.log('TCN err',e.message)}
const s2l=n=>{n.encode(seq);const ctx={h:[...n.encoder.h],c:[...n.encoder.c]};n.decoder.reset();n.decoder.h=[...ctx.h];n.decoder.c=[...ctx.c];let prev=[0],L=0;tg.forEach((t,i)=>{const h=n.decoder.predict(prev);const o=n._project(h);L+=sq(o,t);prev=t});return L};
try{gradcheck('Seq2Seq',()=>new N.Seq2Seq(2,3,1),s2l,(n,l)=>n.trainStep(seq,tg,l))}catch(e){console.log('S2S err',e.message)}
// Adam→SGD temporal para medir gradiente exacto
N.Adam.prototype.step=function(w,g,lr){return w+lr*g};
const ce=(n,tk,t,nc)=>{const o=n.predict(tk);let L=0;for(let i=0;i<tk.length;i++){const p=N.softmax(o.slice(i*nc,i*nc+nc));for(let c=0;c<nc;c++)if(t[i*nc+c]>0)L-=Math.log(p[c])}return L};
const tk=[0,1,2],tt=[1,0,0,1,1,0];
gradcheck('NetworkTransformer',()=>new N.NetworkTransformer(3,{vocabSize:4,d_model:4,nHeads:2,d_ff:6,nBlocks:1,nClasses:2}),n=>ce(n,tk,tt,2),(n,l)=>n.train(tk,tt,l));
const qs=[[.1,.2],[.3,-.1],[.5,.4]];
gradcheck('NetworkTransformerRL',()=>new N.NetworkTransformerRL(3,2,{d_model:4,nHeads:2,d_ff:6,nBlocks:1,nActions:2}),n=>sq(n.predict(qs),[.3,.7]),(n,l)=>n.train(qs,[.3,.7],l));
N.Adam.prototype.step=realAdam;
console.log('\n== B. ¿Aprende? pérdida inicio→fin (mediana de 5 semillas)');
const med=a=>[...a].sort((x,y)=>x-y)[a.length>>1];
function learn(name,fn){const r=[];for(const s of [11,22,33,44,55]){seed(s);r.push(fn())}const a=med(r.map(x=>x[0])),b=med(r.map(x=>x[1]));console.log(`${name.padEnd(36)} ${a.toFixed(4)} → ${b.toFixed(4)}  ${b<a*.7?'aprende':b>a*1.05||!isFinite(b)?'EMPEORA':'NO aprende'}`)}
const XOR=[[0,0],[0,1],[1,0],[1,1]],XT=[0,1,1,0];
learn('Network XOR',()=>{const n=new N.Network(2,4,1);const L=()=>XOR.reduce((s,x,i)=>s+(n.predict(x)[0]-XT[i])**2,0)/4;const a=L();for(let e=0;e<5000;e++)XOR.forEach((x,i)=>n.train(x,XT[i],.5));return[a,L()]});
learn('NetworkN XOR',()=>{const n=new N.NetworkN([2,6,1]);const L=()=>XOR.reduce((s,x,i)=>s+(n.predict(x)[0]-XT[i])**2,0)/4;const a=L();for(let e=0;e<5000;e++)XOR.forEach((x,i)=>n.train(x,[XT[i]],.5));return[a,L()]});
learn('NetworkN Adam XOR',()=>{const n=new N.NetworkN([2,6,1],{optimizer:()=>new N.Adam()});const L=()=>XOR.reduce((s,x,i)=>s+(n.predict(x)[0]-XT[i])**2,0)/4;const a=L();for(let e=0;e<1500;e++)XOR.forEach((x,i)=>n.train(x,[XT[i]],.02));return[a,L()]});
learn('NetworkLSTM secuencia',()=>{const n=new N.NetworkLSTM(2,6,[6,1]);const L=()=>{n.resetState();const v=seq.reduce((s,x,t)=>s+(n.predict(x)[0]-tg[t][0])**2,0)/3;return v};const a=L();n.resetState();for(let e=0;e<1500;e++){n.resetState();seq.forEach(x=>n.predict(x));n.train(tg,.3)}return[a,L()]});
learn('NetworkTransformer copiar tokens',()=>{const n=new N.NetworkTransformer(4,{vocabSize:4,d_model:16,nHeads:2,d_ff:32,nBlocks:1,nClasses:4});const data=[[0,1,2,3],[3,2,1,0],[1,1,2,2],[0,3,0,3]];const oh=t=>t.flatMap(v=>[0,1,2,3].map(c=>c===v?1:0));const L=()=>data.reduce((s,t)=>s+ce(n,t,oh(t),4)/4,0)/4;const a=L();for(let e=0;e<300;e++)data.forEach(t=>n.train(t,oh(t),.005));return[a,L()]});
learn('NetworkTransformerRL regresión',()=>{const n=new N.NetworkTransformerRL(3,2,{d_model:8,nHeads:2,d_ff:16,nBlocks:1,nActions:2});const L=()=>N.mse(n.predict(qs),[.3,.7]);const a=L();for(let e=0;e<300;e++)n.train(qs,[.3,.7],.005);return[a,L()]});
learn('RNN',()=>{const n=new N.RNN(2,6,1);const a=n.backward(seq,tg,0);for(let e=0;e<500;e++)n.backward(seq,tg,.02);return[a,n.backward(seq,tg,0)]});
learn('TCN',()=>{const n=new N.TCN(2,6,1,2,2);const a=n.train(seq,tg,0);for(let e=0;e<300;e++)n.train(seq,tg,.01);return[a,n.train(seq,tg,0)]});
learn('Seq2Seq',()=>{const n=new N.Seq2Seq(2,6,1);const a=n.trainStep(seq,tg,0);for(let e=0;e<300;e++)n.trainStep(seq,tg,.01);return[a,n.trainStep(seq,tg,0)]});
const AED=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
learn('Autoencoder',()=>{const n=new N.Autoencoder(4,[8],3,[8]);const L=()=>AED.reduce((s,x)=>s+N.mse(n.reconstruct(x),x),0)/4;const a=L();for(let e=0;e<8000;e++)AED.forEach(x=>n.train(x,.5));return[a,L()]});
console.log('\n== C. GAN y VAE');
for(const s of [1,2,3]){seed(s);const g=new N.GAN(2,[8],1,[8]);const real=()=>[0.2+(Math.random()-.5)*.02];let b0=0;for(let i=0;i<200;i++)b0+=g.generate()[0]/200;for(let e=0;e<8000;e++)g.trainStep([real()],.05);let b=0;for(let i=0;i<200;i++)b+=g.generate()[0]/200;console.log(`GAN objetivo=0.20  media generada ${b0.toFixed(3)} → ${b.toFixed(3)}`)}
for(const s of [1,2]){seed(s);const v=new N.VAE(4,[8],2,[8]);const D=AED;let t=[];for(let e=0;e<8000;e++)t=D.map(x=>v.train(x,.1));const mus=D.map(x=>v.encode(x).mu.map(m=>+m.toFixed(3)));const rec=D.reduce((a,x)=>a+N.mse(v.decode(v.encode(x).mu),x),0)/4;console.log(`VAE mu por entrada ${JSON.stringify(mus)} recon=${rec.toFixed(4)} (baseline media=0.1875) logVar0=${v.encode(D[0]).logVar.map(x=>x.toFixed(2))}`)}
console.log('\n== D. Sondas puntuales');
seed(9);let n=new N.NetworkN([2,2,1]);n.setWeights([1,2,3]);console.log('NetworkN.setWeights corto → predict',n.predict([1,1]));
n=new N.NetworkN([2,2,1]);N.ModelSaver.fromJSON(n,JSON.stringify({weights:[1]}));console.log('ModelSaver.fromJSON pesos incompletos → predict',n.predict([1,1]));
const l=new N.LSTMLayer(2,2);const snap=l.getWeights();const before=snap.forgetGate.W[0][0];l.predict([.1,.2]);l.backprop([[1,1]],.1);console.log('LSTM getWeights() es referencia viva:',before!==snap.forgetGate.W[0][0]);
const net2=new N.Network(2,3,2);const o0=[...net2.outputLayer.neurons[1].weights];for(let i=0;i<100;i++)net2.train([1,0],1,.5);console.log('Network con 2 salidas: 2ª salida entrenada:',o0.some((v,i)=>v!==net2.outputLayer.neurons[1].weights[i]));
let v=new N.Value(1);try{for(let i=0;i<20000;i++)v=v.add(1);v.backward();console.log('Tape 20k nodos ok')}catch(e){console.log('Tape 20k nodos:',e.message)}
const c2=new N.Conv2D(4,4,1,1,2,{padding:'same'});console.log("Conv2D 'same' kernel par, entrada 4x4 → salida",c2.outputShape());
try{const c1=new N.Conv1D(4,2,1,1,'same');console.log("Conv1D 'same' kernel par, entrada 4 → salida",c1.getOutputLength(),c1.forward([1,2,3,4])[0].length)}catch(e){console.log('Conv1D same',e.message)}
const tr=new N.Trainer(new N.NetworkN([2,2,1]),{epochs:3,earlyStopping:{patience:1,minDelta:0}});const w=console.warn;let warned=0;console.warn=()=>warned++;tr.train({inputs:[[0,0]],targets:[[0]]});console.warn=w;console.log('Trainer earlyStopping sin validación: avisos=',warned,'motivo=',tr.getStopReason());
seed(5);const big=new N.NetworkN([100,256,10],{optimizer:()=>new N.Adam()});const inp=Array(100).fill(.5),tgt=Array(10).fill(.5);let t0=performance.now();for(let i=0;i<200;i++)big.train(inp,tgt,.001);console.log('Rendimiento NetworkN [100,256,10] Adam ms/muestra:',((performance.now()-t0)/200).toFixed(2));
seed(5);const tf=new N.NetworkTransformer(81,{});const tk2=Array(81).fill(1),t2=Array(81*9).fill(0);t0=performance.now();tf.train(tk2,t2,.001);console.log('Rendimiento Transformer por defecto (81 tokens) ms/muestra:',(performance.now()-t0).toFixed(0));
