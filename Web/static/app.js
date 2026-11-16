async function api(url, method="GET", data){
  const opts = { method, headers: {} };
  if (data){ opts.headers["Content-Type"]="application/json"; opts.body=JSON.stringify(data); }
  const res = await fetch(url, opts);
  if(!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json();
}
const $ = (sel)=>document.querySelector(sel);
const convList = $("#conv-list");
const messages = $("#messages");
const header = $("#chat-header");
const msgInput = $("#msg");

let currentToken = null;

async function loadConversations(){
  const { items } = await api("/api/conversations");
  convList.innerHTML = "";
  for(const c of items){
    const li = document.createElement("li");
    li.dataset.token = c.token;
    li.innerHTML = `<span class="title">${escapeHtml(c.title || "שיחה")}</span><span class="meta">${new Date(c.updated_at).toLocaleString()}</span>`;
    li.onclick = ()=>selectConversation(c.token, c.title);
    convList.appendChild(li);
  }
  markActive();
}

async function selectConversation(token, title){
  currentToken = token;
  header.textContent = title || "שיחה";
  const { messages: hist } = await api(`/api/conversations/${token}/history`);
  renderHistory(hist);
  markActive();
}

function markActive(){
  for(const li of convList.querySelectorAll("li")){
    li.classList.toggle("active", li.dataset.token === currentToken);
  }
}

function renderHistory(hist) {
  messages.innerHTML = "";
  for (const m of hist) {
    const div = document.createElement("div");
    div.className = "msg " + (m.role || "assistant");

    if (m.role === "system") {
      div.className = "msg system";
      div.textContent = m.text;
    } else {
      // Replace tabs with spaces for better display
      const formattedText = m.text.replace(/\t/g, '    ');

      // Use marked.js to parse markdown
      const html = marked.parse(formattedText);

      // Create a temporary container to manipulate the HTML
      const temp = document.createElement('div');
      temp.innerHTML = html;

      // Ensure <pre> and <code> blocks have proper formatting
      const codeBlocks = temp.querySelectorAll('code, pre');
      codeBlocks.forEach(block => {
        block.style.whiteSpace = 'pre';
        block.style.direction = 'ltr';
        block.style.textAlign = 'left';
        block.style.display = 'block';
      });

      div.innerHTML = temp.innerHTML;
    }

    messages.appendChild(div);
  }
  messages.scrollTop = messages.scrollHeight;
}

async function newConversation(){
  const created = await api("/api/conversations/new","POST");
  await loadConversations();
  const li = convList.querySelector(`li[data-token="${created.token}"]`);
  const title = li ? li.querySelector(".title").textContent : "בוט מסלול";
  await selectConversation(created.token, title);
}

async function sendMessage(){
  const text = msgInput.value.trim();
  if(!text) return;
  appendLocal("user", text);
  msgInput.value = "";

  // Show loading indicator
  const loadingDiv = showLoading();

  try{
    const payload = { text, token: currentToken };
    const res = await api("/api/send","POST", payload);

    if(!currentToken) currentToken = res.token;

    appendLocal("assistant", res.text);

    await loadConversations();
    markActive();
  }catch(err){
    appendLocal("assistant", "[Error] "+err.message);
  }finally{
    // Remove loading indicator
    if(loadingDiv && loadingDiv.parentNode){
      loadingDiv.parentNode.removeChild(loadingDiv);
    }
  }
}

function showLoading(){
  const div = document.createElement("div");
  div.className = "msg loading";
  const spinner = document.createElement("div");
  spinner.className = "loading-spinner";
  const text = document.createElement("span");
  text.textContent = "חושב...";
  div.appendChild(spinner);
  div.appendChild(text);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function appendLocal(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;

  // First, replace tabs with spaces for better display
  const formattedText = text.replace(/\t/g, '    ');

  // Use marked.js to parse markdown, but ensure newlines are preserved
  const html = marked.parse(formattedText);

  // Create a temporary container to manipulate the HTML
  const temp = document.createElement('div');
  temp.innerHTML = html;

  // Ensure <pre> and <code> blocks have proper formatting
  const codeBlocks = temp.querySelectorAll('code, pre');
  codeBlocks.forEach(block => {
    block.style.whiteSpace = 'pre';
    block.style.direction = 'ltr';
    block.style.textAlign = 'left';
    block.style.display = 'block';
  });

  div.innerHTML = temp.innerHTML;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(s){
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
}

$("#new-conv").onclick = newConversation;
$("#logout").onclick = async ()=>{ await api("/api/logout","POST"); location.href="/login"; };
$("#send-form").addEventListener("submit", (e)=>{ e.preventDefault(); sendMessage(); });

(async function(){
  try{
    await loadConversations();
    const first = convList.querySelector("li");
    if(first){ selectConversation(first.dataset.token, first.querySelector(".title").textContent); }
  }catch(err){ location.href="/login"; }
})();