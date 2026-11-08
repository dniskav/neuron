import fs from 'node:fs'; import path from 'node:path'; import {createRequire} from 'node:module'; import {spawnSync} from 'node:child_process';
const [,,R,S]=process.argv; const require=createRequire(R+'/'); const esbuild=require('esbuild');
const md=fs.readFileSync(R+'/README.md','utf8'); const lines=md.split('\n'); let h='',inb=false,buf=[],start=0; const blocks=[];
lines.forEach((l,i)=>{ if(!inb&&/^#{1,4} /.test(l))h=l.replace(/^#+ /,''); if(/^```(ts|typescript|js|javascript)\s*$/.test(l)){inb=true;buf=[];start=i+1;return} if(inb&&/^```/.test(l)){inb=false;blocks.push({h,start,code:buf.join('\n')});return} if(inb)buf.push(l)});
let ok=0,fail=0; for(const [i,b] of blocks.entries()){ const code=b.code.replace(/from\s+['"]@dniskav\/neuron['"]/g,`from '${S}/dist/index.mjs'`); let js; try{js=esbuild.transformSync(code,{loader:'ts',format:'esm'}).code}catch(e){console.log(`L${b.start} [${b.h}] SYNTAX: ${e.message.split('\n')[1]||e.message}`);fail++;continue}
 const f=`${S}/readme/b${i}.mjs`; fs.writeFileSync(f,js); const r=spawnSync('node',[f],{timeout:30000,encoding:'utf8',cwd:S+'/readme'}); if(r.status===0){ok++;console.log(`OK   L${b.start} [${b.h}]`)}else{fail++;const err=(r.stderr||'').split('\n').find(x=>/Error|error:/.test(x))||(r.error?.code)||'unknown';console.log(`FAIL L${b.start} [${b.h}] ${err}`)}}
console.log(`\nbloques=${blocks.length} ok=${ok} fail=${fail}`);
