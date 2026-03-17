




const chatMessages   = document.getElementById('chat-messages');
const welcomeScreen  = document.getElementById('welcome-screen');
const queryInput     = document.getElementById('query-input');
const sendBtn        = document.getElementById('send-btn');
const clearPromptBtn = document.getElementById('clear-prompt-btn');
const clearChatBtn   = document.getElementById('clear-chat-btn');
const sidebar        = document.getElementById('sidebar');
const uploadArea     = document.getElementById('upload-area');
const fileInput      = document.getElementById('file-input');
const docList        = document.getElementById('doc-list');
const chunkCount     = document.getElementById('chunk-count');
const modelName      = document.getElementById('model-name');
const aboutToggle    = document.getElementById('about-toggle');
const aboutModal     = document.getElementById('about-modal');
const aboutClose     = document.getElementById('about-close');

const settingsToggle = document.getElementById('settings-toggle');
const settingsModal  = document.getElementById('settings-modal');
const settingsClose  = document.getElementById('settings-close');
const settingsSaveBtn = document.getElementById('settings-saveBtn');
const settingLlm     = document.getElementById('setting-llm');
const settingEmbedding = document.getElementById('setting-embedding');
const settingSearch  = document.getElementById('setting-search');
const settingSourcePreference = document.getElementById('setting-source-preference');
const settingOpenaiKey = document.getElementById('setting-openai-key');
const settingGeminiKey = document.getElementById('setting-gemini-key');
const settingOllamaUrl = document.getElementById('setting-ollama-url');
const settingLlmModel = document.getElementById('setting-llm-model');
const settingEmbeddingModel = document.getElementById('setting-embedding-model');
const btnRefreshOllama = document.getElementById('refresh-ollama-models');

const uploadModal    = document.getElementById('upload-modal');
const uploadFilename = document.getElementById('upload-filename');
const uploadProgress = document.getElementById('upload-progress');
const uploadStatus   = document.getElementById('upload-status');

const webSearchToggle = document.getElementById('web-search-toggle');
const webSearchModal  = document.getElementById('web-search-modal');
const webSearchClose  = document.getElementById('web-search-close');
const webSearchInput  = document.getElementById('web-search-query');
const webSearchBtn    = document.getElementById('web-search-btn');
const webSearchResults= document.getElementById('web-search-results');
const webSearchIngestBtn = document.getElementById('web-search-ingest-btn');
const webSearchStatus = document.getElementById('web-search-status');
const sourcePreferenceSelect = document.getElementById('source-preference');

const toastContainer = document.getElementById('toast-container');

let isProcessing = false;

let currentWebSearchResults = [];
let displayedWebSearchResults = [];
let webSearchSortMode = 'citations';

let conversationHistory = [];

const chatWebSearchResultsMap = {};




const API_BASE = '';  // Empty = same origin (works both locally and deployed)

const DEMO_PROMPT = 'Using the indexed papers, when should I choose a transformer over an RNN for sequence modeling, and what trade-offs do the sources highlight?';

const DEMO_MATERIALS = [
  {
    type: 'PDF',
    title: 'Attention Is All You Need',
    note: 'Core transformer paper used for long-range dependency and parallel training claims.',
  },
  {
    type: 'PDF',
    title: 'Sequence to Sequence Learning with Neural Networks',
    note: 'Baseline encoder-decoder framing for recurrent sequence models.',
  },
  {
    type: 'PDF',
    title: 'Learning Phrase Representations using RNN Encoder-Decoder',
    note: 'Explains recurrent modeling strengths and practical sequence trade-offs.',
  },
];

const DEMO_SOURCES = [
  { origin: 'Indexed PDF', label: 'Attention Is All You Need', source_type: 'paper' },
  { origin: 'Indexed PDF', label: 'Sequence to Sequence Learning with Neural Networks', source_type: 'paper' },
  { origin: 'Indexed PDF', label: 'Learning Phrase Representations using RNN Encoder-Decoder', source_type: 'paper' },
];

const DEMO_ANSWER_MARKDOWN = `**Recommendation:** choose a transformer when the task depends on long-range context, you can train offline on enough data, and throughput matters. Choose an RNN when the setup is smaller, more sequential, or needs lightweight step-by-step inference.

**What the demo materials show:**

- **Transformers** win when attention over the full sequence is valuable, because they model distant token interactions directly and train in parallel.
- **RNNs** remain reasonable when sequence state evolves incrementally, latency is tight, or the system needs a simpler recurrent architecture.
- Across these papers, the main trade-off is **context coverage and training efficiency** versus **sequential simplicity and smaller-footprint inference**.

**Bottom line:** if the workload looks like modern document understanding, translation at scale, or long-context reasoning, the indexed materials support using a transformer. If you are building a constrained or streaming system with modest sequence lengths, an RNN can still be the better engineering choice.`;


function getSuggestionChipsHtml() {
  return `
    <div class="suggestions">
      <button class="suggestion-chip" onclick="useSuggestion(this)">
        What are the key findings on attention mechanisms?
      </button>
      <button class="suggestion-chip" onclick="useSuggestion(this)">
        Summarize the main contributions of this paper
      </button>
      <button class="suggestion-chip" onclick="useSuggestion(this)">
        Compare transformer and RNN architectures
      </button>
      <button class="suggestion-chip" onclick="useSuggestion(this)">
        Write research notes on self-supervised learning
      </button>
    </div>
  `;
}


function getDemoSourcesHtml(sources) {
  const listItems = sources.map((source, index) => {
    const label = escapeHtml(source.label || `Source ${index + 1}`);
    const origin = escapeHtml(source.origin || 'Indexed Source');
    const sourceType = escapeHtml(source.source_type || 'document');

    return `
      <li>
        <span class="source-origin">${origin}</span>
        <span>${label}</span>
        <span class="source-kind">${sourceType}</span>
      </li>
    `;
  }).join('');

  return `
    <div class="assistant-sources">
      <div class="assistant-sources-title">Sources Used</div>
      <ul class="assistant-sources-list">${listItems}</ul>
    </div>
  `;
}


function getWelcomeScreenHtml(showDemo = true) {
  if (!showDemo) {
    return `
      <span class="welcome-icon">RA</span>
      <h2>Research Assistant</h2>
      <p>
        Ingest your own PDFs, notes, or web material, then ask questions and get cited answers from the multi-agent workflow.
      </p>
      ${getSuggestionChipsHtml()}
    `;
  }

  const materialCards = DEMO_MATERIALS.map((material) => `
    <div class="welcome-demo-material-card">
      <div class="welcome-demo-material-type">${escapeHtml(material.type)}</div>
      <div class="welcome-demo-material-title">${escapeHtml(material.title)}</div>
      <div class="welcome-demo-material-note">${escapeHtml(material.note)}</div>
    </div>
  `).join('');

  return `
    <span class="welcome-icon">RA</span>
    <h2>Research Assistant</h2>
    <p>
      Demo mode shows what the system looks like after source material has been indexed and a cited answer has already been produced.
    </p>
    <div class="welcome-demo-panel">
      <div class="welcome-demo-header">
        <span class="welcome-demo-tag">Demo Mode</span>
        <div class="welcome-demo-actions">
          <button class="welcome-demo-action" type="button" onclick="setDemoPrompt()">Use in composer</button>
          <button class="welcome-demo-action" type="button" onclick="clearDemoMode()">Clear demo</button>
        </div>
      </div>
      <div class="welcome-demo-section-label">Example Indexed Materials</div>
      <div class="welcome-demo-materials">${materialCards}</div>
      <div class="welcome-demo-chat">
        <div class="message user-message welcome-demo-message">
          <div class="message-avatar">You</div>
          <div class="message-content">
            <div class="message-bubble">${escapeHtml(DEMO_PROMPT)}</div>
          </div>
        </div>
        <div class="message assistant-message welcome-demo-message">
          <div class="message-avatar">RA</div>
          <div class="message-content">
            <div class="message-bubble">${renderMarkdown(DEMO_ANSWER_MARKDOWN)}</div>
            ${getDemoSourcesHtml(DEMO_SOURCES)}
          </div>
        </div>
      </div>
    </div>
    ${getSuggestionChipsHtml()}
  `;
}




function renderMarkdown(text) {
  if (!text) return '';

  let html = text;

  const mathBlocks = [];

  html = html.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
    let rendered = match; // fallback: show raw if KaTeX not loaded
    if (typeof katex !== 'undefined') {
      try { rendered = katex.renderToString(math.trim(), { displayMode: true, throwOnError: false }); }
      catch (_) {}
    }
    const idx = mathBlocks.length;
    mathBlocks.push(rendered);
    return `%%MATH_${idx}%%`;
  });

  html = html.replace(/\$([^\$\n]{1,300}?)\$/g, (match, math) => {
    let rendered = match;
    if (typeof katex !== 'undefined') {
      try { rendered = katex.renderToString(math.trim(), { displayMode: false, throwOnError: false }); }
      catch (_) {}
    }
    const idx = mathBlocks.length;
    mathBlocks.push(rendered);
    return `%%MATH_${idx}%%`;
  });

  html = html.replace(/\\boxed\{([\s\S]*?)\}/g, (_, inner) =>
    `<div class="response-box">${inner}</div>`
  );
  html = html.replace(/(?<![\\a-zA-Z])boxed\{([\s\S]*?)\}/g, (_, inner) =>
    `<div class="response-box">${inner}</div>`
  );

  const codeBlocks = [];
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const index = codeBlocks.length;
    codeBlocks.push(`<pre><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`);
    return `%%CODEBLOCK_${index}%%`;
  });

  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  html = html.replace(/^(?:---|\*\*\*)$/gm, '<hr>');

  html = html.replace(/\n\n+/g, '</p><p>');
  html = html.replace(/<p>(<h[1-6]|<ul|<ol|<pre|<hr|%%CODEBLOCK|%%MATH)/g, '$1');
  html = html.replace(/(<\/h[1-6]>|<\/ul>|<\/ol>|<\/pre>|%%CODEBLOCK_\d+%%|%%MATH_\d+%%)<\/p>/g, '$1');

  html = html.replace(/\n/g, '<br>');

  codeBlocks.forEach((block, i) => {
    html = html.replace(`%%CODEBLOCK_${i}%%`, block);
  });

  mathBlocks.forEach((block, i) => {
    html = html.replace(`%%MATH_${i}%%`, block);
  });

  return html;
}


function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}




function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icons = { success: '✅', error: '⚠️', info: 'ℹ️' };
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> ${escapeHtml(message)}`;

  toastContainer.appendChild(toast);

  setTimeout(() => toast.remove(), 3500);
}





function addUserMessage(text) {
  if (welcomeScreen) {
    welcomeScreen.classList.add('hidden');
  }

  const messageDiv = document.createElement('div');
  messageDiv.className = 'message user-message';
  messageDiv.innerHTML = `
    <div class="message-avatar">You</div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
  `;

  chatMessages.appendChild(messageDiv);
  scrollToBottom();
}


function addAssistantMessage(text) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message assistant-message';
  messageDiv.innerHTML = `
    <div class="message-avatar">RA</div>
    <div class="message-content">
      <div class="message-bubble">${renderMarkdown(text)}</div>
    </div>
  `;

  chatMessages.appendChild(messageDiv);
  scrollToBottom();
  return messageDiv;
}

function renderMessageSources(messageDiv, sources) {
  if (!messageDiv || !Array.isArray(sources) || sources.length === 0) return;

  const messageContent = messageDiv.querySelector('.message-content');
  if (!messageContent) return;

  const sourceBlock = document.createElement('div');
  sourceBlock.className = 'assistant-sources';

  const listItems = sources.map((source, index) => {
    const label = escapeHtml(source.label || source.source || `Source ${index + 1}`);
    const origin = escapeHtml(source.origin || 'Source');
    const sourceType = escapeHtml(source.source_type || 'unknown');
    const url = (source.url || '').trim();

    const linkHtml = url
      ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
      : `<span>${label}</span>`;

    return `
      <li>
        <span class="source-origin">${origin}</span>
        ${linkHtml}
        <span class="source-kind">${sourceType}</span>
      </li>
    `;
  }).join('');

  sourceBlock.innerHTML = `
    <div class="assistant-sources-title">Sources Used</div>
    <ul class="assistant-sources-list">${listItems}</ul>
  `;

  messageContent.appendChild(sourceBlock);
  scrollToBottom();
}


function showThinking() {
  const thinkingDiv = document.createElement('div');
  thinkingDiv.className = 'message assistant-message';
  thinkingDiv.id = 'thinking-indicator';
  thinkingDiv.innerHTML = `
    <div class="message-avatar">RA</div>
    <div class="message-content">
      <div class="thinking-indicator live-run-panel">
        <div class="live-run-header">
          <span class="live-run-title">Live Run</span>
          <span class="live-run-state">Running</span>
        </div>
        <div class="live-run-current" id="live-run-current">Preparing the workflow</div>
        <div class="live-run-list" id="live-run-list"></div>
      </div>
    </div>
  `;

  chatMessages.appendChild(thinkingDiv);
  scrollToBottom();
  return thinkingDiv;
}


function hideThinking() {
  const thinking = document.getElementById('thinking-indicator');
  if (thinking) thinking.remove();
}

function agentKey(name) {
  return String(name || 'agent').toLowerCase().replace(/[^a-z0-9]+/g, '-');
}

function upsertLiveRunItem(panel, agent, state, message = '') {
  if (!panel) return;
  const list = panel.querySelector('#live-run-list');
  const current = panel.querySelector('#live-run-current');
  const stateLabel = panel.querySelector('.live-run-state');

  if (current && message) current.textContent = message;
  if (stateLabel) stateLabel.textContent = state === 'done' ? 'Finalizing' : 'Running';
  if (!list) return;

  const key = agentKey(agent);
  let row = list.querySelector(`[data-agent="${key}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'live-run-item';
    row.dataset.agent = key;
    list.appendChild(row);
  }

  row.className = `live-run-item ${state}`;
  row.innerHTML = `
    <span class="live-run-item-mark">${state === 'done' ? '✓' : '·'}</span>
    <span class="live-run-item-name">${escapeHtml(agent)}</span>
    <span class="live-run-item-text">${escapeHtml(message || (state === 'done' ? 'Completed' : 'Working'))}</span>
  `;
  scrollToBottom();
}

function handleStreamedError(detail) {
  hideThinking();

  if (detail && detail.type === 'embedding_dimension_mismatch') {
    const msgEl = addAssistantMessage(
      `Warning: **Embedding model mismatch detected.**\n\n${detail.message}`
    );
    const actionDiv = document.createElement('div');
    actionDiv.className = 'dimension-mismatch-action';
    actionDiv.innerHTML = `<button class="btn btn-danger btn-sm" id="inline-reset-btn">Reset Collection Now</button>`;
    if (msgEl) {
      const bubble = msgEl.querySelector('.message-bubble') || msgEl;
      bubble.appendChild(actionDiv);
      actionDiv.querySelector('#inline-reset-btn').addEventListener('click', resetCollection);
    }
    return;
  }

  const message = typeof detail === 'string' ? detail : (detail && detail.message) || 'Unknown error';
  addAssistantMessage(`Error: ${message}\n\nPlease check that the backend is running and try again.`);
  showToast(message, 'error');
}

function summarizeStepMessage(step) {
  if (!step || !step.detail) return 'Completed';
  const raw = String(step.detail)
    .replace(/\*\*/g, '')
    .replace(/`/g, '')
    .replace(/\n+/g, ' | ')
    .replace(/\s+/g, ' ')
    .trim();
  return raw.length > 180 ? `${raw.slice(0, 177)}...` : raw;
}


function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}


function renderWelcomeScreen(showDemo = true) {
  chatMessages.innerHTML = '';

  const welcome = document.createElement('div');
  welcome.className = 'welcome-screen';
  welcome.id = 'welcome-screen';
  if (!showDemo) {
    welcome.classList.add('welcome-screen-blank');
  }
  welcome.innerHTML = getWelcomeScreenHtml(showDemo);

  chatMessages.appendChild(welcome);
}


function renderDemoConversation() {
  chatMessages.innerHTML = '';
  conversationHistory = [];
  
  // Render user message
  const userMsg = document.createElement('div');
  userMsg.className = 'message user-message';
  userMsg.innerHTML = `
    <div class="message-avatar">You</div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(DEMO_PROMPT)}</div>
    </div>
  `;
  chatMessages.appendChild(userMsg);
  
  // Render assistant message with sources
  const assistantMsg = document.createElement('div');
  assistantMsg.className = 'message assistant-message';
  const renderedAnswer = renderMarkdown(DEMO_ANSWER_MARKDOWN);
  assistantMsg.innerHTML = `
    <div class="message-avatar">RA</div>
    <div class="message-content">
      <div class="message-bubble">${renderedAnswer}</div>
      ${getDemoSourcesHtml(DEMO_SOURCES)}
    </div>
  `;
  chatMessages.appendChild(assistantMsg);
  
  scrollToBottom();
}


function updatePromptControls() {
  if (!clearPromptBtn) return;
  clearPromptBtn.classList.toggle('is-hidden', !queryInput.value.trim());
}


function clearDemoMode() {
  renderWelcomeScreen(false);
  conversationHistory = [];
  clearPromptInput();
}

window.clearDemoMode = clearDemoMode;


function clearPromptInput() {
  queryInput.value = '';
  autoResizeTextarea();
  updatePromptControls();
  queryInput.focus();
}





async function sendQuery(questionText) {
  if (isProcessing || !questionText.trim()) return;

  isProcessing = true;
  sendBtn.disabled = true;

  addUserMessage(questionText);

  queryInput.value = '';
  queryInput.style.height = 'auto';
  updatePromptControls();
  const thinkingEl = showThinking();

  try {
    const response = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: questionText,
        history: conversationHistory,
        source_preference: sourcePreferenceSelect ? sourcePreferenceSelect.value : 'auto',
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      if (errData.detail) {
        handleStreamedError(errData.detail);
        return;
      }
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalData = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);

        if (event.type === 'status') {
          upsertLiveRunItem(thinkingEl, event.agent, 'running', event.message || 'Working');
        } else if (event.type === 'step') {
          upsertLiveRunItem(
            thinkingEl,
            event.agent || (event.step && event.step.agent),
            'done',
            summarizeStepMessage(event.step)
          );
        } else if (event.type === 'final') {
          finalData = event.data;
        } else if (event.type === 'error') {
          handleStreamedError(event.detail);
          return;
        }
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      if (event.type === 'final') {
        finalData = event.data;
      } else if (event.type === 'error') {
        handleStreamedError(event.detail);
        return;
      }
    }

    hideThinking();
    if (finalData && finalData.steps && finalData.steps.length > 0) {
      renderAgentSteps(finalData.steps);
    }
    if (finalData && finalData.answer) {
      const assistantMessageEl = addAssistantMessage(finalData.answer);
      if (finalData.sources && finalData.sources.length > 0) {
        renderMessageSources(assistantMessageEl, finalData.sources);
      }
      conversationHistory.push({ role: 'user', content: questionText });
      conversationHistory.push({ role: 'assistant', content: finalData.answer });
      if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);
    }

    if (finalData && finalData.web_search_results && finalData.web_search_results.length > 0) {
      renderChatWebSearchResults(finalData.web_search_results);
    }
  } catch (error) {
    handleStreamedError({ message: error.message });
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}


async function resetCollection() {
  if (!confirm(
    'This will permanently delete all indexed vectors.\n' +
    'Your document list will remain but all files will need to be re-ingested.\n\n' +
    'Continue?'
  )) return;

  try {
    showToast('Resetting collection…', 'info');
    const response = await fetch(`${API_BASE}/reset-collection`, { method: 'POST' });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || 'Reset failed');
    showToast(data.message || 'Collection reset. Please re-ingest your documents.', 'success');
    fetchStats();
  } catch (err) {
    showToast(`Reset failed: ${err.message}`, 'error');
  }
}


async function uploadFile(file) {
  uploadModal.classList.add('active');
  uploadFilename.textContent = file.name;
  uploadProgress.style.width = '10%';
  uploadStatus.textContent = 'Uploading...';

  try {
    const formData = new FormData();
    formData.append('file', file);

    uploadProgress.style.width = '40%';
    uploadStatus.textContent = 'Processing document...';

    const response = await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      body: formData,
    });

    uploadProgress.style.width = '80%';

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || 'Upload failed');
    }

    const data = await response.json();
    
    uploadProgress.style.width = '100%';
    uploadStatus.textContent = data.message;

    addDocumentToList(file.name, data.chunks_ingested);
    showToast(`Ingested ${data.chunks_ingested} chunks from ${file.name}`, 'success');

    fetchStats();

    setTimeout(() => {
      uploadModal.classList.remove('active');
      uploadProgress.style.width = '0%';
    }, 1500);

  } catch (error) {
    uploadStatus.textContent = `❌ Error: ${error.message}`;
    showToast(`Upload failed: ${error.message}`, 'error');

    setTimeout(() => {
      uploadModal.classList.remove('active');
      uploadProgress.style.width = '0%';
    }, 2500);
  }
}



async function fetchStats() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (response.ok) {
      const data = await response.json();
      chunkCount.textContent = `${data.stats.total_chunks} (${data.stats.doc_count} docs)`;
      
      const providerMap = {
        'gemini': 'Gemini 2.0 Flash',
        'openai': 'GPT-4o',
        'ollama': 'Ollama Local'
      };
      
      modelName.textContent = providerMap[data.llm_provider] || data.llm_provider;
      
      if (settingLlm) settingLlm.value = data.settings.llm_provider || 'gemini';
      if (settingEmbedding) settingEmbedding.value = data.settings.embedding_provider || 'gemini';
      if (settingSearch) settingSearch.value = data.settings.search_strategy || 'hybrid';
      if (settingSourcePreference && sourcePreferenceSelect) {
        const pref = data.settings.source_preference || 'auto';
        settingSourcePreference.value = pref;
        sourcePreferenceSelect.value = pref;
      }
      if (settingOllamaUrl) settingOllamaUrl.value = data.settings.ollama_base_url || 'http://localhost:11434';
      if (settingOpenaiKey) settingOpenaiKey.value = data.settings.openai_api_key || '';
      if (settingGeminiKey) settingGeminiKey.value = data.settings.google_api_key || '';
      
      toggleModelSettingsVisibility();
      
      await loadModelsFromStats(data.settings);
      
    }
  } catch (err) {
    chunkCount.textContent = 'Offline';
    modelName.textContent = '-';
    console.error("Failed to fetch stats:", err);
  }
}

async function fetchModelsForProvider(provider, settings) {
  if (provider !== 'ollama' && provider !== 'gemini') return [];
  try {
    const response = await fetch(`${API_BASE}/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        provider: provider, 
        base_url: provider === 'ollama' ? (settingOllamaUrl.value || settings.ollama_base_url) : null 
      })
    });
    const data = await response.json();
    return data.models || [];
  } catch (e) {
    console.error(`Failed to load models for ${provider}:`, e);
    return [];
  }
}

async function loadModelsFromStats(settings) {
  const llmProv = settingLlm ? settingLlm.value : settings.llm_provider;
  const embedProv = settingEmbedding ? settingEmbedding.value : settings.embedding_provider;
  
  if (settingLlmModel) {
    const models = await fetchModelsForProvider(llmProv, settings);
    populateDropdown(settingLlmModel, models, settings.llm_model);
  }
  
  if (settingEmbeddingModel) {
    const models = await fetchModelsForProvider(embedProv, settings);
    populateDropdown(settingEmbeddingModel, models, settings.embedding_model);
  }
}

function populateDropdown(selectElement, models, activeModel) {
  if (!selectElement) return;
  selectElement.innerHTML = '';
  
  if (!models || models.length === 0) {
     selectElement.innerHTML = '<option value="">No models found or network error</option>';
     return;
  }
  
  models.forEach(model => {
      const opt = document.createElement('option');
      opt.value = model; opt.textContent = model;
      if (model === activeModel) opt.selected = true;
      selectElement.appendChild(opt);
  });
}

function toggleModelSettingsVisibility() {
  const needsModelLLM = settingLlm && (settingLlm.value === 'ollama' || settingLlm.value === 'gemini');
  const needsModelEmbed = settingEmbedding && (settingEmbedding.value === 'ollama' || settingEmbedding.value === 'gemini');
  const displayVal = (needsModelLLM || needsModelEmbed) ? 'block' : 'none';
  
  document.querySelectorAll('.model-select-setting').forEach(el => {
      el.style.display = displayVal;
  });
}


async function fetchDocuments() {
  try {
    const response = await fetch(`${API_BASE}/documents`);
    if (response.ok) {
      const docs = await response.json();
      docList.innerHTML = ''; // Clear current
      docs.forEach(doc => {
        addDocumentToList(doc.source, doc.chunk_count, doc.status);
      });
    }
  } catch (err) {
    console.error("Failed to fetch documents:", err);
  }
}


async function toggleDocumentStatus(source) {
  try {
    const response = await fetch(`${API_BASE}/documents/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: source }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "Failed to toggle document");
    showToast(data.message || 'Document updated', 'success');
    
    await fetchDocuments();
    await fetchStats();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

window.toggleDocumentStatus = toggleDocumentStatus;



async function deleteDocument(source) {
  if (!confirm(`Are you sure you want to permanently delete "${source}" from the database?`)) {
    return;
  }
  
  try {
    const response = await fetch(`${API_BASE}/documents/delete?source=${encodeURIComponent(source)}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to delete document");
    }
    
    showToast(`Document deleted`, 'success');
    
    await fetchDocuments();
    await fetchStats();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

window.deleteDocument = deleteDocument;




async function searchWeb() {
  const query = webSearchInput.value.trim();
  if (!query) return;

  webSearchBtn.disabled = true;
  webSearchBtn.textContent = 'Searching...';
  webSearchResults.innerHTML = '<p class="search-placeholder">Searching Google Scholar with multiple query variants… please wait.</p>';
  webSearchStatus.textContent = '';
  webSearchIngestBtn.disabled = true;
  webSearchIngestBtn.textContent = 'Ingest Selected (0)';
  currentWebSearchResults = [];
  displayedWebSearchResults = [];

  try {
    const response = await fetch(`${API_BASE}/web-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, max_results: 5 }),
    });

    if (!response.ok) throw new Error("Search request failed");

    const data = await response.json();

    if (data.error) {
      webSearchResults.innerHTML = `<p class="search-placeholder" style="color: var(--error)">Error: ${escapeHtml(data.error)}</p>`;
      return;
    }

    if (!data.results || data.results.length === 0) {
      webSearchResults.innerHTML = '<p class="search-placeholder">No results found.</p>';
      return;
    }

    currentWebSearchResults = data.results;
    webSearchSortMode = 'citations';
    renderWebSearchResults(currentWebSearchResults);

  } catch (err) {
    webSearchResults.innerHTML = `<p class="search-placeholder" style="color: var(--error)">Failed to connect to search API.</p>`;
    showToast(`Search error: ${err.message}`, 'error');
  } finally {
    webSearchBtn.disabled = false;
    webSearchBtn.textContent = 'Search';
  }
}


const REPUTABLE_VENUES = [
  'nature','science','cell','pnas','nejm','jama','lancet',
  'ieee','acm','neurips','nips','icml','iclr','cvpr','eccv','iccv',
  'aaai','ijcai','acl','emnlp','naacl','miccai','kdd','icdm',
];


function sortResults(results, mode) {
  const sorted = [...results];
  if (mode === 'citations') {
    sorted.sort((a, b) => (b.citations || 0) - (a.citations || 0));
  } else if (mode === 'date') {
    sorted.sort((a, b) => (parseInt(b.pub_year) || 0) - (parseInt(a.pub_year) || 0));
  } else if (mode === 'venue') {
    const venueScore = r => {
      const v = (r.venue || '').toLowerCase();
      return REPUTABLE_VENUES.some(kw => v.includes(kw)) ? 1 : 0;
    };
    sorted.sort((a, b) => venueScore(b) - venueScore(a) || (b.citations || 0) - (a.citations || 0));
  }
  return sorted;
}


function buildPaperCard(r, cbId, index, onChangeFn) {
  const card = document.createElement('div');
  card.className = 'paper-result-card';

  const pdfBadge = r.pdf_url
    ? `<a href="${escapeHtml(r.pdf_url)}" target="_blank" class="pdf-badge">PDF</a>`
    : '';

  card.innerHTML = `
    <div class="checkbox-container">
      <input type="checkbox" id="${cbId}" data-index="${index}" data-title="${escapeHtml(r.title)}" onchange="${onChangeFn}">
    </div>
    <div class="paper-content">
      <div class="paper-title"><label for="${cbId}">${escapeHtml(r.title)}</label></div>
      <div class="paper-meta">
        ${escapeHtml(r.authors)} · ${escapeHtml(r.venue)} (${r.pub_year || '?'})
        · <span class="citation-badge">${r.citations || 0} cites</span>
        ${r.url ? `· <a href="${escapeHtml(r.url)}" target="_blank" style="color: var(--accent);">Link</a>` : ''}
        ${pdfBadge}
      </div>
      <div class="paper-abstract">${escapeHtml(r.abstract)}</div>
    </div>
  `;
  return card;
}

function renderWebSearchResults(results) {
  const checkedTitles = new Set(
    [...webSearchResults.querySelectorAll('input[type="checkbox"]:checked')]
      .map(cb => cb.dataset.title)
  );

  const sorted = sortResults(results, webSearchSortMode);
  displayedWebSearchResults = sorted;

  const sortBar = document.getElementById('web-search-sort-bar');
  if (sortBar) {
    sortBar.style.display = 'flex';
    sortBar.querySelectorAll('.sort-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.sort === webSearchSortMode);
    });
  }

  webSearchResults.innerHTML = '';

  sorted.forEach((r, index) => {
    const card = buildPaperCard(r, `paper-cb-${index}`, index, 'updateWebSearchSelection()');
    if (checkedTitles.has(r.title)) {
      card.querySelector('input[type="checkbox"]').checked = true;
    }
    webSearchResults.appendChild(card);
  });

  updateWebSearchSelection();
}


window.setModalSort = function(mode) {
  webSearchSortMode = mode;
  renderWebSearchResults(currentWebSearchResults);
};

function updateWebSearchSelection() {
  const checkboxes = webSearchResults.querySelectorAll('input[type="checkbox"]:checked');
  const count = checkboxes.length;
  webSearchIngestBtn.textContent = `Ingest Selected (${count})`;
  webSearchIngestBtn.disabled = count === 0;
}

window.updateWebSearchSelection = updateWebSearchSelection;

async function ingestSelectedWebDocs() {
  const checkboxes = webSearchResults.querySelectorAll('input[type="checkbox"]:checked');
  if (checkboxes.length === 0) return;

  webSearchIngestBtn.disabled = true;
  webSearchIngestBtn.textContent = 'Ingesting...';
  webSearchClose.disabled = true;
  
  let successCount = 0;
  let failCount = 0;

  for (const cb of checkboxes) {
    const idx = parseInt(cb.getAttribute('data-index'));
    const paper = displayedWebSearchResults[idx];
    
    try {
      webSearchStatus.textContent = `Ingesting ${successCount + failCount + 1}/${checkboxes.length}: ${paper.title.substring(0, 30)}…`;

      let response;
      if (paper.pdf_url) {
        webSearchStatus.textContent += ' (downloading PDF…)';
        response = await fetch(`${API_BASE}/ingest/from-url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: paper.pdf_url, title: paper.title }),
        });
      } else {
        const textContent = `Title: ${paper.title}\nAuthors: ${paper.authors}\nVenue: ${paper.venue} (${paper.pub_year})\nCitations: ${paper.citations}\nURL: ${paper.url}\n\nAbstract:\n${paper.abstract}`;
        response = await fetch(`${API_BASE}/ingest/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            text: textContent,
            title: paper.title,
            source_url: paper.url || `Web: ${paper.title}`
          }),
        });
      }

      if (!response.ok) throw new Error("Failed");
      successCount++;
    } catch(err) {
      failCount++;
      console.error("Failed to ingest paper:", paper.title, err);
    }
  }

  webSearchStatus.textContent = `Done. ${successCount} ingested${failCount ? `, ${failCount} failed` : ''}.`;
  webSearchIngestBtn.textContent = `Ingest Selected (${checkboxes.length})`;
  webSearchIngestBtn.disabled = false;
  webSearchClose.disabled = false;
  
  if (successCount > 0) {
      showToast(`Successfully ingested ${successCount} papers`, 'success');
      await fetchDocuments();
      await fetchStats();
      setTimeout(() => webSearchModal.classList.remove('active'), 1500);
  } else {
      showToast(`Failed to ingest selected papers.`, 'error');
  }
}



window.renderChatWebSearchResults = function(results) {
  const uniqueId = Date.now().toString();

  const sorted = sortResults(results, 'citations');
  chatWebSearchResultsMap[uniqueId] = sorted;
  
  const container = document.createElement('div');
  container.className = 'message assistant-message';
  
  let cardsHtml = '';
  sorted.forEach((r, index) => {
    const cbId = `chat-paper-cb-${uniqueId}-${index}`;
    const pdfBadge = r.pdf_url
      ? `<a href="${escapeHtml(r.pdf_url)}" target="_blank" class="pdf-badge">PDF</a>`
      : '';
    cardsHtml += `
      <div class="paper-result-card" style="margin-bottom: var(--space-sm);">
        <div class="checkbox-container">
          <input type="checkbox" id="${cbId}" class="chat-paper-cb-${uniqueId}"
            data-index="${index}" data-title="${escapeHtml(r.title)}"
            onchange="updateChatWebSearchSelection('${uniqueId}')">
        </div>
        <div class="paper-content">
          <div class="paper-title"><label for="${cbId}">${escapeHtml(r.title)}</label></div>
          <div class="paper-meta">
            ${escapeHtml(r.authors)} · ${escapeHtml(r.venue)} (${r.pub_year || '?'})
            · <span class="citation-badge">${r.citations || 0} cites</span>
            ${r.url ? `· <a target="_blank" href="${escapeHtml(r.url)}" style="color: var(--accent);">Link</a>` : ''}
            ${pdfBadge}
          </div>
          <div class="paper-abstract">${escapeHtml(r.abstract)}</div>
        </div>
      </div>
    `;
  });

  container.innerHTML = `
    <div class="message-avatar">Web</div>
    <div class="message-content">
      <div class="message-bubble" style="background: transparent; border: none; padding: 0;">
        <h4 style="margin-top: 0; margin-bottom: var(--space-md); color: var(--text-primary);">Search Results
          <span style="font-size:0.72rem; font-weight:400; color: var(--text-muted); margin-left: var(--space-xs);">sorted by citations</span>
        </h4>
        <div class="web-search-results-area" style="max-height: none; padding: 0; border: none; background: transparent; overflow-y: visible;">
          ${cardsHtml}
        </div>
        <div style="margin-top: var(--space-md); display: flex; align-items: center; gap: var(--space-md);">
          <button class="btn btn-success" id="chat-web-ingest-btn-${uniqueId}" disabled
            onclick="ingestChatWebDocs('${uniqueId}')">Ingest Selected Papers (0)</button>
          <span id="chat-web-status-${uniqueId}" class="setting-hint" style="margin: 0;"></span>
        </div>
      </div>
    </div>
  `;
  
  chatMessages.appendChild(container);
  scrollToBottom();
}

window.updateChatWebSearchSelection = function(uniqueId) {
  const checkboxes = document.querySelectorAll(`.chat-paper-cb-${uniqueId}:checked`);
  const btn = document.getElementById(`chat-web-ingest-btn-${uniqueId}`);
  if (btn) {
    btn.textContent = `Ingest Selected Papers (${checkboxes.length})`;
    btn.disabled = checkboxes.length === 0;
  }
};

window.ingestChatWebDocs = async function(uniqueId) {
  const checkboxes = document.querySelectorAll(`.chat-paper-cb-${uniqueId}:checked`);
  if (checkboxes.length === 0) return;
  
  const results = chatWebSearchResultsMap[uniqueId];
  if (!results) return;

  const btn = document.getElementById(`chat-web-ingest-btn-${uniqueId}`);
  const statusSpan = document.getElementById(`chat-web-status-${uniqueId}`);
  
  btn.disabled = true;
  btn.textContent = 'Ingesting...';
  
  let successCount = 0;
  let failCount = 0;

  for (const cb of checkboxes) {
    const idx = parseInt(cb.getAttribute('data-index'));
    const paper = results[idx];
    
    try {
      statusSpan.textContent = `Ingesting ${successCount + failCount + 1}/${checkboxes.length}…`;

      let response;
      if (paper.pdf_url) {
        statusSpan.textContent += ' (downloading PDF…)';
        response = await fetch(`${API_BASE}/ingest/from-url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: paper.pdf_url, title: paper.title }),
        });
      } else {
        const textContent = `Title: ${paper.title}\nAuthors: ${paper.authors}\nVenue: ${paper.venue} (${paper.pub_year})\nCitations: ${paper.citations}\nURL: ${paper.url}\n\nAbstract:\n${paper.abstract}`;
        response = await fetch(`${API_BASE}/ingest/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            text: textContent,
            title: paper.title,
            source_url: paper.url || `Web: ${paper.title}`
          }),
        });
      }

      if (!response.ok) throw new Error("Failed");
      successCount++;
    } catch(err) {
      failCount++;
      console.error("Failed to ingest paper:", paper.title, err);
    }
  }

  statusSpan.textContent = `Done. ${successCount} ingested${failCount ? `, ${failCount} failed` : ''}.`;
  btn.textContent = `Ingest Selected Papers (${checkboxes.length})`;
  btn.disabled = false;
  
  if (successCount > 0) {
      showToast(`Successfully ingested ${successCount} papers`, 'success');
      await fetchDocuments();
      await fetchStats();
  } else {
      showToast(`Failed to ingest selected papers.`, 'error');
  }
};




function renderAgentSteps(steps) {
  if (!steps || steps.length === 0) return;

  const panel = document.createElement('div');
  panel.className = 'message assistant-message';

  let stepsHtml = '';
  steps.forEach((step, i) => {
    const stepId = `step-${Date.now()}-${i}`;
    const detail = renderMarkdown(step.detail || '');
    const thinkingBadge = step.thinking
      ? '<span class="thinking-badge">Thinking model</span>'
      : '';

    const reasoningHtml = step.reasoning
      ? `<div class="agent-step-reasoning">
           <div class="agent-step-reasoning-header" onclick="toggleStepReasoning('${stepId}-reasoning')">
             Chain of Thought <span id="${stepId}-reasoning-chevron">▶</span>
           </div>
           <div class="agent-step-reasoning-body" id="${stepId}-reasoning">${renderMarkdown(step.reasoning)}</div>
         </div>`
      : '';

    stepsHtml += `
      <div class="agent-step">
        <div class="agent-step-header" onclick="toggleStepDetail('${stepId}')">
          <span class="agent-step-icon">${i + 1}</span>
          <span class="agent-step-name">${escapeHtml(step.agent)}</span>
          ${thinkingBadge}
          <span class="agent-step-status">done</span>
          <span class="agent-step-chevron" id="${stepId}-chevron">▶</span>
        </div>
        <div class="agent-step-detail" id="${stepId}">
          ${detail}
          ${reasoningHtml}
        </div>
      </div>
    `;
  });

  panel.innerHTML = `
    <div class="message-avatar message-avatar-neutral">Run</div>
    <div class="message-content">
      <div class="agent-steps-panel">
        <div class="agent-steps-header" onclick="toggleAgentPanel(this)">
          <span>Agent Pipeline — ${steps.length} step${steps.length !== 1 ? 's' : ''}</span>
          <span class="agent-steps-toggle">▼</span>
        </div>
        <div class="agent-steps-list">
          ${stepsHtml}
        </div>
      </div>
    </div>
  `;

  chatMessages.appendChild(panel);
  scrollToBottom();
}

window.toggleAgentPanel = function(header) {
  const list = header.nextElementSibling;
  const toggle = header.querySelector('.agent-steps-toggle');
  list.classList.toggle('open');
  toggle.textContent = list.classList.contains('open') ? '▲' : '▼';
};

window.toggleStepDetail = function(stepId) {
  const detail = document.getElementById(stepId);
  const chevron = document.getElementById(`${stepId}-chevron`);
  if (detail) {
    detail.classList.toggle('open');
    if (chevron) chevron.textContent = detail.classList.contains('open') ? '▼' : '▶';
  }
};

window.toggleStepReasoning = function(id) {
  const body = document.getElementById(id);
  const chevronId = id + '-chevron';
  const chevron = document.getElementById(chevronId);
  if (body) {
    body.classList.toggle('open');
    if (chevron) chevron.textContent = body.classList.contains('open') ? '▼' : '▶';
  }
};





function addDocumentToList(name, chunks, status = "active") {
  const li = document.createElement('li');
  li.className = status === 'active' ? 'doc-item' : 'doc-item disabled';
  const toggleIcon = status === 'active' ? '☑' : '☐';

  li.innerHTML = `
    <button class="doc-toggle-btn doc-toggle-left" title="Toggle Document" onclick="toggleDocumentStatus('${escapeHtml(name)}')">${toggleIcon}</button>
    <span class="doc-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
    <span class="doc-chunks">${chunks} chunks</span>
    <div class="doc-item-actions">
        <button class="doc-delete-btn" title="Delete Document" onclick="deleteDocument('${escapeHtml(name)}')">❌</button>
    </div>
  `;

  docList.appendChild(li);
}




function autoResizeTextarea() {
  queryInput.style.height = 'auto';                // Reset height
  queryInput.style.height = queryInput.scrollHeight + 'px'; // Set to content height
}




function useSuggestion(element) {
  const text = element.textContent.trim();
  queryInput.value = text;
  updatePromptControls();
  sendQuery(text);
}

window.useSuggestion = useSuggestion;




sendBtn.addEventListener('click', () => {
  sendQuery(queryInput.value);
});

if (settingLlm) settingLlm.addEventListener('change', async () => {
    toggleModelSettingsVisibility();
    const models = await fetchModelsForProvider(settingLlm.value, {});
    populateDropdown(settingLlmModel, models, '');
});

if (settingEmbedding) settingEmbedding.addEventListener('change', async () => {
    toggleModelSettingsVisibility();
    const models = await fetchModelsForProvider(settingEmbedding.value, {});
    populateDropdown(settingEmbeddingModel, models, '');
});

if (btnRefreshOllama) {
  btnRefreshOllama.addEventListener('click', async () => {
    btnRefreshOllama.textContent = 'Fetching...';
    try {
        if (settingLlm && settingLlm.value === 'ollama' && settingLlmModel) {
             const models = await fetchModelsForProvider('ollama', {});
             populateDropdown(settingLlmModel, models, settingLlmModel.value);
        }
        if (settingEmbedding && settingEmbedding.value === 'ollama' && settingEmbeddingModel) {
             const models = await fetchModelsForProvider('ollama', {});
             populateDropdown(settingEmbeddingModel, models, settingEmbeddingModel.value);
        }
        showToast(`Tested connection & refreshed models`, 'success');
    } catch(err) {
        showToast("Error connecting to Ollama URL", "error");
    } finally {
        btnRefreshOllama.textContent = 'Test Connection';
    }
  });
}
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();  // Prevent the Enter from adding a newline
    sendQuery(queryInput.value);
  }
});

queryInput.addEventListener('input', () => {
  autoResizeTextarea();
  updatePromptControls();
});

if (clearPromptBtn) {
  clearPromptBtn.addEventListener('click', clearPromptInput);
}

clearChatBtn.addEventListener('click', () => {
  renderDemoConversation();
  clearPromptInput();
  showToast('Demo restored', 'info');
});

function openSettingsModal() {
  if (settingsModal) settingsModal.classList.add('active');
}

function closeSettingsModal() {
  if (settingsModal) settingsModal.classList.remove('active');
}

function openAboutModal() {
  if (aboutModal) aboutModal.classList.add('active');
}

function closeAboutModal() {
  if (aboutModal) aboutModal.classList.remove('active');
}

if (settingsToggle) {
  settingsToggle.addEventListener('click', openSettingsModal);
}

if (settingsClose) {
  settingsClose.addEventListener('click', closeSettingsModal);
}

if (settingsModal) {
  settingsModal.addEventListener('click', (event) => {
    if (event.target === settingsModal) {
      closeSettingsModal();
    }
  });
}

if (aboutToggle) {
  aboutToggle.addEventListener('click', openAboutModal);
}

if (aboutClose) {
  aboutClose.addEventListener('click', closeAboutModal);
}

if (aboutModal) {
  aboutModal.addEventListener('click', (event) => {
    if (event.target === aboutModal) {
      closeAboutModal();
    }
  });
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeSettingsModal();
    closeAboutModal();
    if (webSearchModal) webSearchModal.classList.remove('active');
  }
});

if (settingsSaveBtn) {
  settingsSaveBtn.addEventListener('click', async () => {
    const llmProvider = settingLlm.value || 'gemini';
    const embedProvider = settingEmbedding.value || 'gemini';
    
    let llmModel = settingLlmModel.value;
    if (!llmModel) {
        llmModel = llmProvider === 'gemini' ? 'gemini-2.0-flash' : 'qwen3.5:latest';
    }
    
    let embedModel = settingEmbeddingModel.value;
    if (!embedModel) {
        embedModel = embedProvider === 'gemini' ? 'models/gemini-embedding-001' : 'nomic-embed-text';
    }

    const payload = {
      llm_provider: llmProvider,
      embedding_provider: embedProvider,
      search_strategy: settingSearch.value,
      source_preference: settingSourcePreference ? settingSourcePreference.value : (sourcePreferenceSelect ? sourcePreferenceSelect.value : 'auto'),
      openai_api_key: settingOpenaiKey.value || null,
      google_api_key: settingGeminiKey.value || null,
      ollama_base_url: settingOllamaUrl.value || null,
      llm_model: llmModel,
      embedding_model: embedModel
    };
    
    settingsSaveBtn.textContent = 'Saving...';
    try {
      const resp = await fetch(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error('Settings update failed');
      showToast('Settings saved successfully', 'success');
      if (data.collection_reset) {
        showToast('Embedding changed. Old indexed documents were removed and need to be re-ingested.', 'info');
      }
      setTimeout(closeSettingsModal, 300);
      fetchStats(); // Update UI
      fetchDocuments();
    } catch (e) {
      showToast(e.message, 'error');
    } finally {
      settingsSaveBtn.textContent = 'Save Settings';
    }
  });
}

if (settingSourcePreference && sourcePreferenceSelect) {
  settingSourcePreference.addEventListener('change', () => {
    sourcePreferenceSelect.value = settingSourcePreference.value;
  });
  sourcePreferenceSelect.addEventListener('change', () => {
    settingSourcePreference.value = sourcePreferenceSelect.value;
  });
}

const resetCollectionBtn = document.getElementById('reset-collection-btn');
if (resetCollectionBtn) {
  resetCollectionBtn.addEventListener('click', resetCollection);
}

uploadArea.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  const files = e.target.files;
  if (files.length > 0) {
    uploadFile(files[0]);
  }
  fileInput.value = '';
});

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();  // Required to allow dropping
  uploadArea.style.borderColor = 'var(--accent)';
  uploadArea.style.background = 'rgba(124,106,239,0.1)';
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.style.borderColor = '';
  uploadArea.style.background = '';
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = '';
  uploadArea.style.background = '';

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    uploadFile(files[0]);
  }
});

if (webSearchToggle) {
  webSearchToggle.addEventListener('click', () => {
    webSearchModal.classList.add('active');
    setTimeout(() => webSearchInput.focus(), 100);
  });
}

if (webSearchClose) {
  webSearchClose.addEventListener('click', () => {
    webSearchModal.classList.remove('active');
  });
}

if (webSearchBtn) {
  webSearchBtn.addEventListener('click', searchWeb);
}

if (webSearchInput) {
  webSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      searchWeb();
    }
  });
}

if (webSearchIngestBtn) {
  webSearchIngestBtn.addEventListener('click', ingestSelectedWebDocs);
}






const resizeHandle = document.getElementById('resize-handle');

if (resizeHandle) {
  let isResizing = false;

  resizeHandle.addEventListener('mousedown', (e) => {
    isResizing = true;
    resizeHandle.classList.add('dragging');
    document.body.classList.add('resizing');
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const newWidth = e.clientX;
    if (newWidth >= 180 && newWidth <= 520) {
      sidebar.style.width = newWidth + 'px';
    }
  });

  document.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      resizeHandle.classList.remove('dragging');
      document.body.classList.remove('resizing');
    }
  });
}

fetchStats();
fetchDocuments();

renderDemoConversation();
queryInput.focus();

console.log('Research Assistant loaded');
