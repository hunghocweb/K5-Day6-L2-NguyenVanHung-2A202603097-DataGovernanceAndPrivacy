'use strict';
const $=id=>document.getElementById(id);
const stages=[
 ['Ảnh đầu vào','Đọc ảnh, xoay theo EXIF và chuẩn hóa kích thước','source.png',1],
 ['Nhận biết loại ảnh','Phân biệt ảnh chụp thật và ảnh vẽ 2D từ nội dung ảnh','tiles.png',2],
 ['Phát hiện theo nhánh','2D elip và xử lý mờ → 2D chi tiết khi cần; ảnh thật dùng YuNet + YOLO','detections.png',4],
 ['Gộp vùng trùng','Giữ vùng che của nhánh được chọn; gộp các phát hiện trùng','merged.png',4],
 ['Nới vùng che','Chỉ nới bổ sung theo thanh trượt; 0% giữ nguyên vùng của nhánh đã chọn','padding.png',5],
 ['Hoàn tất','Blur trên ảnh gốc và xuất PNG, CSV cùng báo cáo','result.png',6]
];
function stagesFor(mode){return mode&&mode!=='auto'?stages.map((s,i)=>[i===0?'Ảnh đầu vào':i===5?'Kết quả phiên cũ':`Bước ${i+1} của phiên cũ`,'Ảnh trung gian của thuật toán cũ, trước khi gộp luồng tự động.',s[2],s[3]]):stages;}
const routeLabels={real:'Ảnh chụp thật → YuNet + YOLO','2d':'Ảnh vẽ 2D → hình học và nét chữ',both:'Chưa rõ loại ảnh → kiểm tra cả hai nhánh'};
function branchLabel(data){
 if(data?.cascade?.selected==='ellipse_blur_2d')return 'Ảnh 2D → bản elip và xử lý mờ gốc';
 if(data?.cascade?.selected==='detailed_2d')return 'Ảnh 2D → bộ chi tiết dự phòng';
 return data?.routing?routeLabels[data.routing.route]:'';
}
let token,job=null,file=null,localURL=null,available=-1,selected=0,running=false,lastState=null,timer;
function artifact(name,download=false){return `/api/walk/file?job=${job}&name=${name}${download?'&download=1':''}`;}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function choose(f){
 if(running){status('Chờ ảnh đang chạy hoàn tất trước khi chọn ảnh mới.');return;}
 if(!f)return;
 if(f.size>15*1024*1024){status('Ảnh vượt 15 MB. Chọn ảnh nhỏ hơn.',true);return;}
 file=f;job=null;lastState=null;available=0;selected=0;
 if(localURL)URL.revokeObjectURL(localURL);
 localURL=URL.createObjectURL(f);history.replaceState(null,'','/walkthrough');
 $('fileInfo').textContent=`${f.name} · ${(f.size/1024/1024).toFixed(2)} MB`;
 $('summary').hidden=true;$('runProgress').value=0;$('run').disabled=!token;
 $('route').textContent='Loại ảnh: sẽ tự nhận biết khi chạy';
 status('Ảnh đã sẵn sàng. Bấm Chạy xử lý ảnh.');render();
}
function render(){
 const currentStages=stagesFor(lastState?.mode),list=$('steps');list.replaceChildren();
 currentStages.forEach((s,i)=>{
  const b=document.createElement('button');b.className='step'+(i===selected?' active':'');
  b.disabled=i>available;b.setAttribute('aria-current',i===selected?'step':'false');
  const n=document.createElement('span');n.className='step-number';n.textContent=i+1;
  const label=document.createElement('span'),strong=document.createElement('strong'),small=document.createElement('small');
  strong.textContent=s[0];small.textContent=i<=available?'Có thể xem':'Chưa chạy';
  label.append(strong,small);b.append(n,label);
  b.onclick=()=>{$('autoFollow').checked=false;selected=i;render();};list.append(b);
 });
 if(available<0)return;
 const s=currentStages[selected];$('stageTitle').textContent=`${selected+1}. ${s[0]}`;
 $('stageDescription').textContent=(selected===1||selected===2)&&lastState?.routing?branchLabel(lastState):s[1];
 if(selected===3&&lastState?.detector_padding_applied)$('stageDescription').textContent='Bản 2D gốc giữ mức nới có sẵn. Thanh trượt ở bước sau chỉ nới bổ sung.';
 $('stageCounter').textContent=`${selected+1} / 6`;$('empty').hidden=true;
 $('stageImage').hidden=false;$('stageImage').src=job?artifact(s[2]):localURL;
 $('previousStep').disabled=selected===0;$('nextStep').disabled=selected>=available;
}
function complete(data){
 $('pad').value=Math.round(data.padding*100);$('padValue').textContent=Math.round(data.padding*100)+'%';
 $('summary').hidden=false;
 $('counts').textContent=data.stage_summary||`${data.n_face} vùng mặt · ${data.n_plate} vùng biển số · ${data.final_count} vùng che.`;
 const legacy=data.mode!=='auto'?'Phiên cũ, dùng thuật toán trước khi gộp luồng. Tải ảnh lên để chạy phiên mới. ':'';
 $('notice').textContent=legacy+(data.final_count===0?'Không tìm thấy vùng nào; đây không phải xác nhận ảnh không có PII. ':'')+`Ảnh đầu vào ${data.original_size.join(' × ')} → ảnh xuất ${data.working_size.join(' × ')}. `+data.notice;
 $('downloadImage').href=artifact('result.png',true);$('downloadZip').href=artifact('result.zip',true);
 const rows=$('boxRows');rows.replaceChildren();
 data.boxes.forEach(b=>{const row=document.createElement('tr');for(const v of [b.type,b.x,b.y,b.w,b.h,Number(b.confidence).toFixed(3)]){const td=document.createElement('td');td.textContent=v;row.append(td);}rows.append(row);});
}
function enable(){running=false;$('run').disabled=!file||!token;$('file').disabled=false;$('pad').disabled=false;$('run').textContent='Chạy lại ảnh';}
async function poll(){
 try{
  const response=await fetch(`/api/walk/status?job=${job}`),data=await response.json();
  if(!response.ok)throw Error(data.error||'Không đọc được tiến trình.');
  lastState=data;status(data.message,data.state==='error');$('runProgress').value=data.stage||0;
  if(data.routing)$('route').textContent=branchLabel(data);
  else if(data.mode&&data.mode!=='auto')$('route').textContent='Đang xem phiên cũ';
  if(data.state==='error'){enable();return;}
  const next=stages.reduce((n,s,i)=>data.stage>=s[3]?i:n,0);
  if(next!==available){available=next;if($('autoFollow').checked)selected=next;render();}
  if(data.state==='complete'){enable();complete(data);return;}
  if(Date.now()/1000-data.started>300){status('Xử lý quá 5 phút. Xem worker.log trong out/walkthrough.',true);enable();return;}
  timer=setTimeout(poll,700);
 }catch(error){status(error.message+' Bấm chạy lại nếu server đã khởi động.',true);enable();}
}
$('run').onclick=async()=>{
 if(!file||running)return;
 running=true;$('run').disabled=true;$('file').disabled=true;$('pad').disabled=true;
 $('run').textContent='Đang xử lý…';$('summary').hidden=true;lastState=null;
 clearTimeout(timer);status('Đang gửi ảnh tới server trên máy…');
 try{
  const response=await fetch(`/api/walk/upload?mode=auto&padding=${Number($('pad').value)/100}`,{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Review-Token':token},body:file});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Không nhận được ảnh.');
  job=data.job;history.replaceState(null,'','/walkthrough?job='+job);available=0;selected=0;render();poll();
 }catch(error){status(error.message,true);enable();}
};
$('file').onchange=()=>choose($('file').files[0]);
$('pad').oninput=()=>$('padValue').textContent=$('pad').value+'%';
$('previousStep').onclick=()=>{if(selected>0){selected--;$('autoFollow').checked=false;render();}};
$('nextStep').onclick=()=>{if(selected<available){selected++;$('autoFollow').checked=false;render();}};
$('stageImage').onerror=()=>status('Không tải được ảnh bước này. Chờ xử lý hoặc tải lại trang.',true);
for(const name of ['dragenter','dragover'])$('drop').addEventListener(name,e=>{e.preventDefault();$('drop').classList.add('drag');});
for(const name of ['dragleave','drop'])$('drop').addEventListener(name,e=>{e.preventDefault();$('drop').classList.remove('drag');});
$('drop').addEventListener('drop',e=>choose(e.dataTransfer.files[0]));
fetch('/api/state').then(r=>{if(!r.ok)throw Error('Server chưa sẵn sàng.');return r.json();}).then(data=>{
 token=data.token;const saved=new URLSearchParams(location.search).get('job');
 if(saved){job=saved;available=-1;running=true;$('file').disabled=true;$('pad').disabled=true;poll();}
 else{$('run').disabled=!file;status('Chọn ảnh từ máy để bắt đầu.');render();}
}).catch(error=>status(error.message,true));
