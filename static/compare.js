const $=id=>document.getElementById(id);
let catalog=[],dataset=null,uploads=[],demoMode=true,busy=false, demoFiles=[], randomMode=false, sampleSeed=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,options){const r=await fetch(url,options);if(!r.ok){const b=await r.json().catch(()=>({}));throw Error(b.detail||`Request failed (${r.status})`)}return r.json()}
function renderInputs(){
 $('demo-count-label').hidden=!demoMode;
 const inputs=demoMode?demoFiles.map(name=>({name,url:`/api/demos/${dataset.id}/${name}`})):uploads;
 $('source').textContent=demoMode?`Demo scan · ${inputs.length} tiles${randomMode ? ` · random seed ${sampleSeed}` : ''}`:`Your images · ${inputs.length} files`;
 updatePairCost(inputs.length);
 $('images').innerHTML=inputs.map((f,i)=>`<div class="thumb"><img src="${esc(f.url)}" alt="Input ${i+1}" loading="lazy"><div title="${esc(f.name)}">${i+1} · ${esc(f.name.slice(0,15))}</div>${demoMode?'':`<button type="button" data-move="${i}" data-dir="-1" aria-label="Move image ${i+1} earlier">←</button><button type="button" data-move="${i}" data-dir="1" aria-label="Move image ${i+1} later">→</button>`}</div>`).join('');
}
$('images').onclick=e=>{const b=e.target.closest('[data-move]');if(!b)return;let i=+b.dataset.move,j=i+(+b.dataset.dir);if(j<0||j>=uploads.length)return;[uploads[i],uploads[j]]=[uploads[j],uploads[i]];renderInputs()};
$('files').onchange=e=>{uploads.forEach(f=>URL.revokeObjectURL(f.url));uploads=[...e.target.files].map(file=>({file,name:file.name,url:URL.createObjectURL(file)}));demoMode=false;renderInputs()};
function shuffled(items, seed) {
 const out=[...items]; let state=seed>>>0;
 const next=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
 for(let i=out.length-1;i>0;i--){const j=Math.floor(next()*(i+1));[out[i],out[j]]=[out[j],out[i]];}return out;
}
function newSeed(){return crypto.getRandomValues(new Uint32Array(1))[0];}
function updatePairCost(n=demoMode?demoFiles.length:uploads.length){
 const unordered=$('pairing').value==='unordered';$('disconnected-label').hidden=!unordered;
 $('pair-cost').textContent=unordered?`${n*(n-1)/2} pair comparisons. All-pairs search can be slow, especially for dense models. Unrelated images cannot form a panorama.`:`${Math.max(0,n-1)} adjacent pair comparisons.`;
}
function chooseDemo(random=false){demoMode=true;randomMode=random;sampleSeed=random?newSeed():null;demoFiles=(random?shuffled(dataset?.files||[],sampleSeed):dataset?.files||[]).slice(0,+$('demo-count').value);$('pairing').value=random?'unordered':'sequential';renderInputs();}
$('demo').onclick=()=>chooseDemo();$('random').onclick=()=>chooseDemo(true);
$('demo-count').onchange=()=>chooseDemo(randomMode);
$('pairing').onchange=()=>updatePairCost();
$('shuffle').onclick=()=>{sampleSeed=newSeed();if(demoMode){demoFiles=shuffled(demoFiles,sampleSeed);}else{uploads=shuffled(uploads,sampleSeed);}$('pairing').value='unordered';renderInputs();};
$('select-ready').onclick=()=>document.querySelectorAll('[name=engine]').forEach(c=>c.checked=!c.disabled&&c.value!=='mast3r'&&c.value!=='superglue');
$('select-none').onclick=()=>document.querySelectorAll('[name=engine]').forEach(c=>c.checked=false);
function renderResults(run){
 $('status').textContent=run.progress;
 const sum=a=>(a||[]).reduce((x,y)=>x+y,0);
 $('summary').innerHTML=run.results.length?`<table><thead><tr><th>Pipeline</th><th>Status</th><th>Time</th><th>Matches</th><th>Inliers</th></tr></thead><tbody>${run.results.map(r=>`<tr><td>${esc(catalog.find(e=>e.id===r.engine)?.label||r.engine)}</td><td>${esc(r.status)}</td><td>${r.elapsed??'…'} s</td><td>${sum(r.matches)}</td><td>${sum(r.inliers)}</td></tr>`).join('')}</tbody></table>`:'';
 $('results').innerHTML=run.results.map(r=>{const e=catalog.find(e=>e.id===r.engine);return `<article class="result ${esc(r.status)}">${r.image?`<a href="${r.image}" target="_blank" rel="noopener"><img src="${r.image}" alt="${esc(e.label)} stitched panorama"></a>`:''}<div class="body"><h3>${esc(e?.label||r.engine)}</h3>${r.error?`<p>${esc(r.error)}</p>`:`<div class="metrics"><div><strong>${r.elapsed??'…'}</strong><small>seconds</small></div><div><strong>${sum(r.inliers)}</strong><small>inliers / ${sum(r.matches)} matches</small></div></div>`}${r.warning?`<p class="warning">${esc(r.warning)}</p>`:''}${r.output?`<p>${r.output.w} × ${r.output.h} px · <a href="${r.image}" download="${r.engine}.jpg">Download panorama ↗</a></p>`:''}${r.graph?`<details open><summary>Overlap graph · ${r.graph.included.length}/${run.inputs.length} images in largest group</summary><p>Groups (input numbers): ${r.graph.components.map(c=>c.join(', ')).join(' / ')}<br>Accepted overlaps: ${r.graph.accepted_pairs}/${r.graph.tested_pairs}</p><details><summary>Pair diagnostics</summary><p>${r.graph.pairs.map(p=>`${p.pair.join(' ↔ ')}: ${p.inliers}/${p.matches} inliers · ${p.accepted?'accepted':esc(p.reason)}`).join('<br>')}</p></details></details>`:''}<details><summary>Stage timings & pair counts</summary><p>${Object.entries(r.stages).map(([k,v])=>`${esc(k)}: ${v}s`).join(' · ')}</p><p>Matches: ${r.matches.join(', ')||'—'}<br>Inliers: ${r.inliers.join(', ')||'—'}</p></details></div></article>`}).join('');
 if(run.status==='complete'){$('report').href=`/api/comparisons/${run.id}/report`;$('report').hidden=false}
}
async function poll(id){try{const run=await api(`/api/comparisons/${id}`);renderResults(run);if(['queued','running'].includes(run.status)){setTimeout(()=>poll(id),1000);return}localStorage.removeItem('stitch-comparison');}catch(e){$('status').textContent=e.message+' Reload to reconnect.'}busy=false;$('run').disabled=false;$('controls').disabled=false;}
$('form').onsubmit=async event=>{event.preventDefault();if(busy)return;try{const selected=[...document.querySelectorAll('[name=engine]:checked')].map(c=>c.value);if(!selected.length)throw Error('Select at least one pipeline.');if(!demoMode&&uploads.length<2)throw Error('Upload at least two overlapping images.');const form=new FormData();form.append('options',JSON.stringify({engines:selected,engine_options:Object.fromEntries(selected.map(id=>[id,EngineControls.read(document.getElementById('model-options-'+id))])),dataset:dataset?.id,demo_files:demoFiles, sample_seed:sampleSeed, input_test:demoMode?(randomMode?'random-tiles':'demo'):'uploads', alignment:$('alignment').value, pairing:$('pairing').value, disconnected:$('disconnected').value,feature_max_dim:+$('resolution').value,max_keypoints:+$('keypoints').value,crop:$('crop').checked}));if(!demoMode)uploads.forEach(f=>form.append('files',f.file));busy=true;$('run').disabled=true;$('controls').disabled=true;$('report').hidden=true;$('status').textContent='Preparing comparison…';const run=await api('/api/comparisons',{method:'POST',body:form});localStorage.setItem('stitch-comparison',run.id);poll(run.id);}catch(e){$('status').textContent=e.message;busy=false;$('run').disabled=false;$('controls').disabled=false;}};
(async()=>{try{const [c,d]=await Promise.all([api('/api/engines'),api('/api/demos')]);catalog=c.engines;dataset=d.datasets[0];demoFiles=(dataset?.files||[]).slice(0,+$('demo-count').value);$('device').textContent=`● ${c.device.toUpperCase()} execution`;$('engines').innerHTML=catalog.map(e=>`<div class="engine"><label><input type="checkbox" name="engine" value="${e.id}" ${e.available?'':'disabled'} ${['sift','orb','xfeat'].includes(e.id)&&e.available?'checked':''}>${esc(e.label)}</label><p>${esc(e.family)} · ${esc(e.status)}</p><details><summary>License & setup</summary><p>${esc(e.license)} · <a href="${e.source}" target="_blank" rel="noopener">Source ↗</a></p>${e.setup?`<p><code>${esc(e.setup)}</code></p>`:''}</details><div id="model-options-${e.id}"></div>${e.installed&&e.implementation_status!=='blocked'?`<button type="button" data-verify="${e.id}">Verify inference</button><p id="verify-status-${e.id}"></p>`:''}</div>`).join('');for(const entry of catalog){if(entry.implementation_status!=='blocked')EngineControls.render(document.getElementById('model-options-'+entry.id),entry);}renderInputs();const id=localStorage.getItem('stitch-comparison');if(id){busy=true;$('run').disabled=true;$('controls').disabled=true;poll(id)}}catch(e){$('status').textContent=e.message}})();

$('engines').addEventListener('click', async event => {
 const button=event.target.closest('[data-verify]'); if(!button)return;
 const entry=catalog.find(e=>e.id===button.dataset.verify);button.disabled=true;
 try{await EngineControls.verify(entry,EngineControls.read(document.getElementById('model-options-'+entry.id)),document.getElementById('verify-status-'+entry.id));
 const fresh=(await EngineControls.catalog()).find(e=>e.id===entry.id);Object.assign(entry,fresh);
 document.querySelector(`[name=engine][value="${entry.id}"]`).disabled=!entry.available;
 }catch(error){document.getElementById('verify-status-'+entry.id).textContent=error.message;}finally{button.disabled=false;}
});
