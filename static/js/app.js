
let allConversations = [];
let currentFilter = 'all';

// Setup Markdown Parser (Marked.js)
if (window.marked) {
    marked.setOptions({
        highlight: function (code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        langPrefix: 'hljs language-'
    });
}

async function loadStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const stats = await response.json();

        const update = (id, val) => {
            const el = document.getElementById(id);
            if (el) {
                // Animate numbers if they changed
                const current = parseInt(el.textContent.replace(/,/g, '')) || 0;
                if (current !== val) {
                    el.textContent = typeof val === 'number' ? val.toLocaleString() : val;
                    el.classList.add('updated');
                    setTimeout(() => el.classList.remove('updated'), 1000);
                }
            }
        };

        update('total-convs', stats.total_conversations);
        update('total-msgs', stats.total_messages);
        update('total-tokens', stats.total_tokens);

    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

async function loadConversations() {
    try {
        const response = await fetch('/api/conversations');
        allConversations = await response.json();
        renderTable();
    } catch (error) {
        console.error('Error loading conversations:', error);
        const content = document.getElementById('table-content');
        if (content) {
            content.innerHTML = `
                <div class="empty-state">
                    <h3>❌ Error Loading Conversations</h3>
                    <p>${error.message}</p>
                </div>
            `;
        }
    }
}

function renderTable() {
    const content = document.getElementById('table-content');
    const searchInput = document.getElementById('search');

    if (!content) return;

    // Filter logic
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    let filtered = allConversations.filter(conv => {
        // Search filter
        const matchesSearch = !searchTerm ||
            conv.category?.toLowerCase().includes(searchTerm) ||
            conv.agent_a_model?.toLowerCase().includes(searchTerm) ||
            conv.agent_b_model?.toLowerCase().includes(searchTerm) ||
            conv.seed_prompt?.toLowerCase().includes(searchTerm);

        // Status filter
        const matchesStatus = currentFilter === 'all' ||
            (currentFilter === 'error' ? conv.status === 'error' :
                currentFilter === 'completed' ? conv.status === 'completed' :
                    currentFilter === 'running' ? conv.status === 'running' : true);

        return matchesSearch && matchesStatus;
    });

    if (filtered.length === 0) {
        content.innerHTML = `
            <div class="empty-state">
                <h3>📭 No Conversations Found</h3>
                <p>Try adjusting your search or filters.</p>
            </div>
        `;
        return;
    }

    const html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Agents</th>
                    <th>Turns</th>
                    <th>Started</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                ${filtered.map(conv => `
                    <tr onclick="viewConversation(${conv.id})">
                        <td class="font-mono text-secondary">#${conv.id}</td>
                        <td><span class="category-tag">${conv.category || 'N/A'}</span></td>
                        <td>
                            <div class="model-info">
                                <span class="model-a">A: ${conv.agent_a_model}</span>
                                <span class="model-b">B: ${conv.agent_b_model}</span>
                            </div>
                        </td>
                        <td>${conv.total_turns}</td>
                        <td class="text-secondary font-mono">${new Date(conv.start_time).toLocaleString()}</td>
                        <td>
                            <span class="status-badge status-${conv.status}">
                                ${conv.status}
                            </span>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    content.innerHTML = html;
}

async function viewConversation(id) {
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modal-body');
    const modalTitle = document.getElementById('modal-title');
    const modalSubtitle = document.getElementById('modal-subtitle');

    if (!modal || !modalBody) return;

    modal.classList.add('active');
    modalBody.innerHTML = `
        <div class="loading-skeleton">
            <div class="skeleton-row"></div>
            <div class="skeleton-row"></div>
        </div>
    `;

    try {
        const response = await fetch(`/api/conversation/${id}`);
        const data = await response.json();

        const conv = data.conversation;
        const messages = data.messages;

        // Update Title & Subtitle
        if (modalTitle) modalTitle.textContent = `Conversation #${id}`;
        if (modalSubtitle) modalSubtitle.textContent = conv.category;

        // Setup Download Button
        const downloadBtn = document.getElementById('download-json');
        if (downloadBtn) {
            downloadBtn.onclick = (e) => {
                e.stopPropagation();
                downloadConversation(conv, messages);
            };
        }

        modalBody.innerHTML = `
            <div class="conversation-meta">
                <div class="meta-grid">
                    <div class="meta-item">
                        <strong>Status</strong>
                        <span class="status-text ${conv.status === 'completed' ? 'text-green' : ''}">${conv.status}</span>
                    </div>
                    <div class="meta-item">
                        <strong>Total Turns</strong>
                        ${conv.total_turns}
                    </div>
                    <div class="meta-item">
                        <strong>Started</strong>
                        ${new Date(conv.start_time).toLocaleString()}
                    </div>
                    <div class="meta-item">
                        <strong>Duration</strong>
                        ${conv.end_time ? calculateDuration(conv.start_time, conv.end_time) : 'Ongoing'}
                    </div>
                </div>
            </div>

            <div class="seed-prompt">
                <strong>🌱 Seed Prompt</strong>
                <p>${conv.seed_prompt}</p>
            </div>

            <div class="chat-container">
                ${messages.map(msg => renderMessage(msg)).join('')}
            </div>
        `;

        // Trigger Highlight.js on new content
        if (window.hljs) {
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }

    } catch (error) {
        console.error(error);
        modalBody.innerHTML = `
            <div class="empty-state">
                <h3>❌ Error Loading Conversation</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}

function renderMessage(msg) {
    const agentClass = msg.role === 'agent_a' ? 'agent-a' : 'agent-b';
    const agentName = msg.role === 'agent_a' ? 'Agent A' : 'Agent B';
    const agentInitial = msg.role === 'agent_a' ? 'A' : 'B';

    // Render Markdown if available
    const contentHtml = window.marked ? marked.parse(msg.content) : `<p>${msg.content}</p>`;

    return `
        <div class="message ${agentClass}">
            <div class="message-header">
                <div class="message-header-left">
                    <div class="agent-avatar">${agentInitial}</div>
                    <span class="agent-name">${agentName}</span>
                </div>
                <span class="turn-badge">Turn ${msg.turn_number}</span>
            </div>
            <div class="message-content">
                ${contentHtml}
            </div>
            <div class="message-meta">
                <span>${msg.model}</span>
                <span>${msg.token_count || 0} tokens</span>
                <span>T=${msg.temperature}</span>
            </div>
        </div>
    `;
}

function calculateDuration(start, end) {
    const s = new Date(start);
    const e = new Date(end);
    const diffMs = e - s;
    const diffMins = Math.round(diffMs / 60000);
    return `${diffMins} min`;
}

function downloadConversation(conv, messages) {
    const data = {
        metadata: conv,
        messages: messages
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${conv.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function copyConversationJSON(conv, messages, btn) {
    const data = {
        metadata: conv,
        messages: messages
    };
    try {
        await navigator.clipboard.writeText(JSON.stringify(data, null, 2));

        // Visual feedback
        const originalIcon = btn.innerHTML;
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        const originalColor = btn.style.color;
        btn.style.color = 'var(--status-completed-text)';

        setTimeout(() => {
            btn.innerHTML = originalIcon;
            btn.style.color = originalColor;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy:', err);
    }
}

function closeModal() {
    const modal = document.getElementById('modal');
    if (modal) modal.classList.remove('active');
}

function handleImportJSON() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const reader = new FileReader();
            reader.onload = async (event) => {
                try {
                    const data = JSON.parse(event.target.result);

                    // Basic validation: check if it has messages or metadata
                    if (!data.messages && !data.metadata) {
                        throw new Error("Invalid conversation JSON structure");
                    }

                    // If it's a standard export from this tool, it might be nested
                    const payload = data.metadata ? {
                        ...data.metadata,
                        messages: data.messages
                    } : data;

                    const response = await fetch('/api/conversation/create', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (response.ok) {
                        const result = await response.json();
                        alert(`Successfully imported conversation #${result.id}`);
                        loadStatistics();
                        loadConversations();
                    } else {
                        const error = await response.json();
                        alert(`Import failed: ${error.error}`);
                    }
                } catch (err) {
                    alert(`Error parsing JSON: ${err.message}`);
                }
            };
            reader.readAsText(file);
        } catch (err) {
            console.error('File read error:', err);
        }
    };
    input.click();
}

function openCreateModal() {
    const modal = document.getElementById('create-modal');
    if (modal) {
        modal.classList.add('active');
        // Clear previous entries
        document.getElementById('form-messages-list').innerHTML = '';
        document.getElementById('create-form').reset();
        addMessageField(); // Start with one empty message
    }
}

function closeCreateModal() {
    const modal = document.getElementById('create-modal');
    if (modal) modal.classList.remove('active');
}

function addMessageField() {
    const list = document.getElementById('form-messages-list');
    const index = list.children.length;

    const div = document.createElement('div');
    div.className = 'message-entry';
    div.innerHTML = `
        <span class="remove-btn" onclick="this.parentElement.remove()">&times;</span>
        <div class="form-row">
            <div class="form-group">
                <label>Role</label>
                <select name="msg_role_${index}">
                    <option value="agent_a">Agent A</option>
                    <option value="agent_b">Agent B</option>
                </select>
            </div>
            <div class="form-group">
                <label>Model</label>
                <input type="text" name="msg_model_${index}" placeholder="Optional override">
            </div>
        </div>
        <div class="form-group" style="margin-top: 8px;">
            <label>Content</label>
            <textarea name="msg_content_${index}" rows="2" required></textarea>
        </div>
    `;
    list.appendChild(div);
}

async function handleCreateSubmit(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        category: formData.get('category'),
        agent_a_model: formData.get('agent_a_model'),
        agent_b_model: formData.get('agent_b_model'),
        seed_prompt: formData.get('seed_prompt'),
        messages: []
    };

    // Extract messages
    const messageEntries = document.querySelectorAll('.message-entry');
    messageEntries.forEach((entry, i) => {
        data.messages.push({
            role: entry.querySelector('select').value,
            model: entry.querySelector('input').value || (entry.querySelector('select').value === 'agent_a' ? data.agent_a_model : data.agent_b_model),
            content: entry.querySelector('textarea').value,
            turn_number: i + 1
        });
    });

    try {
        const response = await fetch('/api/conversation/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const result = await response.json();
            alert('Conversation created successfully!');
            closeCreateModal();
            loadStatistics();
            loadConversations();
        } else {
            const err = await response.json();
            alert(`Error: ${err.error}`);
        }
    } catch (err) {
        alert(`Failed to save: ${err.message}`);
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Import JSON
    const importBtn = document.getElementById('import-json');
    if (importBtn) {
        importBtn.addEventListener('click', handleImportJSON);
    }

    // New Conversation
    const newBtn = document.getElementById('new-conv');
    if (newBtn) {
        newBtn.addEventListener('click', openCreateModal);
    }

    // Create Form
    const createForm = document.getElementById('create-form');
    if (createForm) {
        createForm.addEventListener('submit', handleCreateSubmit);
    }

    // Search
    const searchInput = document.getElementById('search');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            renderTable();
        });

        // Keyboard shortcut for search
        document.addEventListener('keydown', (e) => {
            if (e.key === '/' && document.activeElement !== searchInput) {
                e.preventDefault();
                searchInput.focus();
            }
        });
    }

    // Filter Chips
    const chips = document.querySelectorAll('.filter-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            // Remove active class from all
            chips.forEach(c => c.classList.remove('active'));
            // Add to clicked
            chip.classList.add('active');
            // Update filter
            currentFilter = chip.dataset.filter;
            renderTable();
        });
    });

    // Modal closing
    const modal = document.getElementById('modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target.id === 'modal') {
                closeModal();
            }
        });

        // Close on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('active')) {
                closeModal();
            }
        });
    }

    // Create Modal closing
    const createModal = document.getElementById('create-modal');
    if (createModal) {
        createModal.addEventListener('click', (e) => {
            if (e.target.id === 'create-modal') {
                closeCreateModal();
            }
        });

        // Close on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && createModal.classList.contains('active')) {
                closeCreateModal();
            }
        });
    }

    // Initial load
    loadStatistics();
    loadConversations();

    // Auto-refresh stats every 10s
    setInterval(loadStatistics, 10000);
    // Refresh table every 30s to catch new convos without being too aggressive
    setInterval(loadConversations, 30000);
});
