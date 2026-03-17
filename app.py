from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import json
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gallery-secret-key-2024'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
DATA_DIR = os.path.join(BASE_DIR, 'data')
ALBUMS_FILE = os.path.join(DATA_DIR, 'albums.json')
MODEL_ID = "Salesforce/blip-image-captioning-base"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load BLIP model once at startup
# ---------------------------------------------------------------------------
_processor = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _processor, _model
    if _processor is None or _model is None:
        logger.info(f"Loading {MODEL_ID} onto {_device}…")
        _processor = BlipProcessor.from_pretrained(MODEL_ID)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(_device)
        logger.info("Model ready.")


# ---------------------------------------------------------------------------
# Album persistence helpers
# ---------------------------------------------------------------------------

def load_albums():
    if not os.path.exists(ALBUMS_FILE):
        return {}
    with open(ALBUMS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_albums(albums):
    with open(ALBUMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(albums, f, indent=2)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def album_upload_dir(album_id):
    path = os.path.join(app.config['UPLOAD_FOLDER'], album_id)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# HuggingFace captioning
# ---------------------------------------------------------------------------

def generate_caption(image_path):
    """Generate caption locally using Salesforce/blip-image-captioning-base."""
    try:
        load_model()
        image = Image.open(image_path).convert('RGB')
        inputs = _processor(image, return_tensors="pt").to(_device)
        with torch.no_grad():
            out = _model.generate(**inputs, max_length=50, num_beams=5)
        caption = _processor.decode(out[0], skip_special_tokens=True)
        return caption if caption else 'No caption generated'
    except Exception as e:
        logger.error(f"Caption error: {e}")
        return f'Caption error: {str(e)}'


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HOME_HTML)


@app.route('/album/<album_id>')
def album_view(album_id):
    albums = load_albums()
    if album_id not in albums:
        return render_template_string(NOT_FOUND_HTML), 404
    return render_template_string(ALBUM_HTML, album_id=album_id)


# --- REST API ---

@app.route('/api/albums', methods=['GET'])
def api_get_albums():
    albums = load_albums()
    result = []
    for alb in albums.values():
        cover = alb['images'][0]['filename'] if alb['images'] else None
        result.append({
            'id': alb['id'],
            'name': alb['name'],
            'description': alb.get('description', ''),
            'image_count': len(alb['images']),
            'cover': cover,
            'created_at': alb['created_at'],
        })
    result.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(result)


@app.route('/api/albums', methods=['POST'])
def api_create_album():
    data = request.get_json()
    if not data or not data.get('name', '').strip():
        return jsonify({'error': 'Album name is required'}), 400
    album_id = str(uuid.uuid4())
    album = {
        'id': album_id,
        'name': data['name'].strip(),
        'description': data.get('description', '').strip(),
        'created_at': datetime.utcnow().isoformat(),
        'images': [],
    }
    albums = load_albums()
    albums[album_id] = album
    save_albums(albums)
    album_upload_dir(album_id)
    return jsonify({'id': album_id, 'name': album['name']}), 201


@app.route('/api/albums/<album_id>', methods=['GET'])
def api_get_album(album_id):
    albums = load_albums()
    if album_id not in albums:
        return jsonify({'error': 'Album not found'}), 404
    return jsonify(albums[album_id])


@app.route('/api/albums/<album_id>', methods=['DELETE'])
def api_delete_album(album_id):
    albums = load_albums()
    if album_id not in albums:
        return jsonify({'error': 'Album not found'}), 404
    # Remove image files
    alb_dir = os.path.join(app.config['UPLOAD_FOLDER'], album_id)
    if os.path.exists(alb_dir):
        import shutil
        shutil.rmtree(alb_dir)
    del albums[album_id]
    save_albums(albums)
    return jsonify({'success': True})


@app.route('/api/albums/<album_id>/upload', methods=['POST'])
def api_upload_images(album_id):
    albums = load_albums()
    if album_id not in albums:
        return jsonify({'error': 'Album not found'}), 404

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    upload_dir = album_upload_dir(album_id)
    added = []
    errors = []

    for file in files:
        if not file or file.filename == '':
            continue
        if not allowed_file(file.filename):
            errors.append(f'{file.filename}: unsupported format')
            continue
        original_name = file.filename
        ext = original_name.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(upload_dir, unique_name)
        file.save(filepath)

        # Generate caption via HuggingFace API
        caption = generate_caption(filepath)

        image_entry = {
            'filename': unique_name,
            'original_name': secure_filename(original_name),
            'caption': caption,
            'uploaded_at': datetime.utcnow().isoformat(),
        }
        albums[album_id]['images'].append(image_entry)
        added.append(image_entry)

    save_albums(albums)
    return jsonify({'added': added, 'errors': errors})


@app.route('/api/albums/<album_id>/images/<filename>/caption', methods=['POST'])
def api_regenerate_caption(album_id, filename):
    """Re-generate the caption for a specific image."""
    albums = load_albums()
    if album_id not in albums:
        return jsonify({'error': 'Album not found'}), 404

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], album_id, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image not found'}), 404

    caption = generate_caption(filepath)

    for img in albums[album_id]['images']:
        if img['filename'] == filename:
            img['caption'] = caption
            break
    save_albums(albums)
    return jsonify({'caption': caption})


@app.route('/api/albums/<album_id>/images/<filename>', methods=['DELETE'])
def api_delete_image(album_id, filename):
    albums = load_albums()
    if album_id not in albums:
        return jsonify({'error': 'Album not found'}), 404

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], album_id, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    albums[album_id]['images'] = [
        img for img in albums[album_id]['images'] if img['filename'] != filename
    ]
    save_albums(albums)
    return jsonify({'success': True})


@app.route('/uploads/<album_id>/<filename>')
def serve_image(album_id, filename):
    directory = os.path.join(app.config['UPLOAD_FOLDER'], album_id)
    return send_from_directory(directory, filename)


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'hf_token_set': bool(app.config['HF_API_TOKEN']),
    })


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

_BASE_STYLES = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f0f2f5;
    min-height: 100vh;
    color: #333;
}
nav {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    position: sticky; top: 0; z-index: 100;
}
nav a { color: white; text-decoration: none; font-weight: 500; }
nav .brand { font-size: 1.4em; font-weight: 700; letter-spacing: -0.5px; }
.nav-actions { display: flex; gap: 12px; align-items: center; }
.btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 10px 20px; border-radius: 8px; border: none;
    font-size: 0.95em; font-weight: 500; cursor: pointer;
    transition: all 0.2s; text-decoration: none;
}
.btn-primary { background: #667eea; color: white; }
.btn-primary:hover { background: #5a67d8; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
.btn-white { background: white; color: #667eea; }
.btn-white:hover { background: #f0f0ff; transform: translateY(-1px); }
.btn-danger { background: #e53e3e; color: white; }
.btn-danger:hover { background: #c53030; }
.btn-sm { padding: 6px 14px; font-size: 0.85em; }
.btn-outline { background: transparent; border: 2px solid #667eea; color: #667eea; }
.btn-outline:hover { background: #667eea; color: white; }
.container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
h1 { font-size: 2em; font-weight: 700; color: #2d3748; }
h2 { font-size: 1.5em; font-weight: 600; color: #2d3748; }
/* Modal */
.modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.5); z-index: 200;
    align-items: center; justify-content: center;
}
.modal-overlay.open { display: flex; }
.modal {
    background: white; border-radius: 16px;
    padding: 32px; max-width: 480px; width: 90%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}
.modal h2 { margin-bottom: 24px; }
.form-group { margin-bottom: 18px; }
.form-group label { display: block; margin-bottom: 6px; font-weight: 500; color: #4a5568; font-size: 0.9em; }
.form-group input, .form-group textarea {
    width: 100%; padding: 10px 14px; border: 2px solid #e2e8f0;
    border-radius: 8px; font-size: 0.95em; transition: border-color 0.2s;
    font-family: inherit;
}
.form-group input:focus, .form-group textarea:focus { outline: none; border-color: #667eea; }
.form-group textarea { resize: vertical; min-height: 80px; }
.modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 24px; }
/* Toast */
.toast {
    position: fixed; bottom: 24px; right: 24px; z-index: 300;
    background: #2d3748; color: white; padding: 14px 20px;
    border-radius: 10px; font-size: 0.9em; display: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2); max-width: 320px;
}
.toast.success { background: #276749; }
.toast.error { background: #9b2c2c; }
.toast.show { display: block; animation: slideIn 0.3s ease; }
@keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
"""

HOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhotoGallery — My Albums</title>
<style>
""" + _BASE_STYLES + """
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 64px 24px; text-align: center;
}
.hero h1 { font-size: 2.8em; font-weight: 800; margin-bottom: 12px; }
.hero p { font-size: 1.15em; opacity: 0.9; max-width: 500px; margin: 0 auto 28px; }
.albums-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.album-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 24px;
}
.album-card {
    background: white; border-radius: 16px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.25s; cursor: pointer; text-decoration: none; color: inherit;
    display: block;
}
.album-card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,0.15); }
.album-cover {
    width: 100%; height: 200px; object-fit: cover;
    background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
    display: flex; align-items: center; justify-content: center;
}
.album-cover img { width: 100%; height: 100%; object-fit: cover; }
.album-cover .no-cover {
    font-size: 3.5em; color: #a0aec0;
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    width: 100%; height: 100%;
    background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
    justify-content: center;
}
.album-info { padding: 20px; }
.album-info h3 { font-size: 1.1em; font-weight: 600; margin-bottom: 6px; color: #2d3748; }
.album-meta { font-size: 0.85em; color: #718096; display: flex; align-items: center; gap: 12px; }
.album-actions { padding: 0 20px 16px; display: flex; gap: 8px; }
.empty-state {
    text-align: center; padding: 80px 24px; color: #a0aec0;
    grid-column: 1 / -1;
}
.empty-state .icon { font-size: 4em; margin-bottom: 16px; }
.empty-state p { font-size: 1.1em; margin-bottom: 24px; }
</style>
</head>
<body>
<nav>
    <a href="/" class="brand">📸 PhotoGallery</a>
    <div class="nav-actions">
        <button class="btn btn-white" onclick="openCreateModal()">+ New Album</button>
    </div>
</nav>

<div class="hero">
    <h1>My Photo Gallery</h1>
    <p>Organize your memories into albums. AI automatically captions every photo.</p>
    <button class="btn btn-white" onclick="openCreateModal()" style="font-size:1.05em; padding:12px 28px;">
        + Create Your First Album
    </button>
</div>

<div class="container">
    <div class="albums-header">
        <h2 id="albumCount">Albums</h2>
        <button class="btn btn-primary" onclick="openCreateModal()">+ New Album</button>
    </div>
    <div class="album-grid" id="albumGrid">
        <div class="empty-state">
            <div class="icon">🖼️</div>
            <p>No albums yet. Create one to get started!</p>
        </div>
    </div>
</div>

<!-- Create Album Modal -->
<div class="modal-overlay" id="createModal">
    <div class="modal">
        <h2>Create New Album</h2>
        <div class="form-group">
            <label>Album Name *</label>
            <input type="text" id="albumName" placeholder="e.g. Summer Vacation 2024" maxlength="80">
        </div>
        <div class="form-group">
            <label>Description (optional)</label>
            <textarea id="albumDesc" placeholder="What's this album about?"></textarea>
        </div>
        <div class="modal-actions">
            <button class="btn btn-outline" onclick="closeCreateModal()">Cancel</button>
            <button class="btn btn-primary" onclick="createAlbum()">Create Album</button>
        </div>
    </div>
</div>

<div class="toast" id="toast"></div>

<script>
async function loadAlbums() {
    const res = await fetch('/api/albums');
    const albums = await res.json();
    const grid = document.getElementById('albumGrid');
    const count = document.getElementById('albumCount');
    count.textContent = albums.length + ' Album' + (albums.length !== 1 ? 's' : '');

    if (!albums.length) {
        grid.innerHTML = `<div class="empty-state">
            <div class="icon">🖼️</div>
            <p>No albums yet. Create one to get started!</p>
            <button class="btn btn-primary" onclick="openCreateModal()">+ New Album</button>
        </div>`;
        return;
    }

    grid.innerHTML = albums.map(a => `
        <div class="album-card" onclick="location.href='/album/${a.id}'">
            <div class="album-cover">
                ${a.cover
                    ? `<img src="/uploads/${a.id}/${a.cover}" alt="${escHtml(a.name)}">`
                    : `<div class="no-cover"><span>📷</span></div>`}
            </div>
            <div class="album-info">
                <h3>${escHtml(a.name)}</h3>
                <div class="album-meta">
                    <span>🖼 ${a.image_count} photo${a.image_count !== 1 ? 's' : ''}</span>
                    <span>${formatDate(a.created_at)}</span>
                </div>
                ${a.description ? `<p style="margin-top:8px;font-size:.85em;color:#718096;">${escHtml(a.description)}</p>` : ''}
            </div>
            <div class="album-actions" onclick="event.stopPropagation()">
                <button class="btn btn-sm btn-primary" onclick="location.href='/album/${a.id}'">Open</button>
                <button class="btn btn-sm btn-danger" onclick="deleteAlbum('${a.id}', event)">Delete</button>
            </div>
        </div>
    `).join('');
}

function escHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatDate(iso) {
    return new Date(iso).toLocaleDateString(undefined, { year:'numeric', month:'short', day:'numeric' });
}

function openCreateModal() {
    document.getElementById('createModal').classList.add('open');
    document.getElementById('albumName').focus();
}

function closeCreateModal() {
    document.getElementById('createModal').classList.remove('open');
    document.getElementById('albumName').value = '';
    document.getElementById('albumDesc').value = '';
}

async function createAlbum() {
    const name = document.getElementById('albumName').value.trim();
    if (!name) { showToast('Please enter an album name', 'error'); return; }
    const desc = document.getElementById('albumDesc').value.trim();
    const res = await fetch('/api/albums', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ name, description: desc })
    });
    if (res.ok) {
        const data = await res.json();
        closeCreateModal();
        showToast('Album created!', 'success');
        location.href = '/album/' + data.id;
    } else {
        showToast('Failed to create album', 'error');
    }
}

async function deleteAlbum(id, e) {
    e.stopPropagation();
    if (!confirm('Delete this album and all its photos?')) return;
    const res = await fetch('/api/albums/' + id, { method: 'DELETE' });
    if (res.ok) { showToast('Album deleted', 'success'); loadAlbums(); }
    else showToast('Delete failed', 'error');
}

function showToast(msg, type='') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast show' + (type ? ' ' + type : '');
    setTimeout(() => t.className = 'toast', 3000);
}

document.getElementById('createModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeCreateModal();
});
document.getElementById('albumName').addEventListener('keydown', e => {
    if (e.key === 'Enter') createAlbum();
});

loadAlbums();
</script>
</body>
</html>"""

ALBUM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhotoGallery — Album</title>
<style>
""" + _BASE_STYLES + """
.page-header {
    background: white; border-bottom: 1px solid #e2e8f0;
    padding: 24px 32px; display: flex; align-items: center; gap: 16px;
}
.page-header .back { color: #667eea; text-decoration: none; font-size: 1.5em; }
.page-header .info h1 { font-size: 1.6em; }
.page-header .info p { color: #718096; font-size: 0.9em; margin-top: 4px; }
.page-header .header-actions { margin-left: auto; display: flex; gap: 10px; }

/* Upload zone */
.upload-section { background: white; border-radius: 16px; padding: 28px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.upload-section h3 { margin-bottom: 16px; color: #4a5568; font-size: 1em; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.drop-zone {
    border: 2.5px dashed #cbd5e0; border-radius: 12px;
    padding: 36px; text-align: center; cursor: pointer;
    transition: all 0.2s; background: #fafafa;
}
.drop-zone:hover, .drop-zone.dragover { border-color: #667eea; background: #f0f0ff; }
.drop-zone .icon { font-size: 2.5em; margin-bottom: 10px; }
.drop-zone p { color: #718096; font-size: 0.95em; }
.drop-zone strong { color: #667eea; }
#fileInput { display: none; }
.upload-progress { margin-top: 16px; display: none; }
.progress-bar-wrap { background: #e2e8f0; border-radius: 99px; height: 8px; overflow: hidden; margin-top: 8px; }
.progress-bar { background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; border-radius: 99px; transition: width 0.3s; width: 0%; }
.progress-text { font-size: 0.85em; color: #718096; margin-top: 6px; }

/* Image grid */
.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 20px;
}
.image-card {
    background: white; border-radius: 14px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: all 0.25s;
}
.image-card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.14); }
.image-thumb {
    width: 100%; height: 200px; object-fit: cover; display: block;
    cursor: pointer;
}
.image-body { padding: 16px; }
.image-caption {
    font-size: 0.9em; line-height: 1.5; color: #4a5568;
    min-height: 42px;
}
.caption-loading { color: #a0aec0; font-style: italic; font-size: 0.9em; }
.image-footer {
    padding: 10px 16px; border-top: 1px solid #f0f0f0;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.8em; color: #a0aec0;
}
.image-footer button { font-size: 0.8em; }

/* Lightbox */
.lightbox {
    display: none; position: fixed; inset: 0; z-index: 400;
    background: rgba(0,0,0,0.92); align-items: center; justify-content: center;
    flex-direction: column;
}
.lightbox.open { display: flex; }
.lightbox img { max-width: 90vw; max-height: 75vh; border-radius: 8px; object-fit: contain; }
.lightbox-caption {
    color: white; margin-top: 20px; font-size: 1em;
    max-width: 700px; text-align: center; line-height: 1.6;
    background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px 20px;
}
.lightbox-close {
    position: absolute; top: 20px; right: 24px;
    color: white; font-size: 2em; cursor: pointer; background: none; border: none;
    opacity: 0.7;
}
.lightbox-close:hover { opacity: 1; }
.lightbox-nav {
    position: absolute; top: 50%; transform: translateY(-50%);
    background: rgba(255,255,255,0.15); border: none; color: white;
    font-size: 2em; width: 52px; height: 52px; border-radius: 50%;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background 0.2s;
}
.lightbox-nav:hover { background: rgba(255,255,255,0.3); }
#lightboxPrev { left: 20px; }
#lightboxNext { right: 20px; }
.empty-state {
    text-align: center; padding: 80px 24px; color: #a0aec0;
    grid-column: 1 / -1;
}
.empty-state .icon { font-size: 4em; margin-bottom: 16px; }
</style>
</head>
<body>
<nav>
    <a href="/" class="brand">📸 PhotoGallery</a>
    <div class="nav-actions">
        <a href="/" class="btn btn-white">← All Albums</a>
    </div>
</nav>

<div class="page-header">
    <div class="info">
        <h1 id="albumTitle">Loading...</h1>
        <p id="albumMeta"></p>
    </div>
    <div class="header-actions">
        <label for="fileInput" class="btn btn-primary">+ Upload Photos</label>
        <input type="file" id="fileInput" multiple accept="image/*">
    </div>
</div>

<div class="container">
    <!-- Upload section -->
    <div class="upload-section">
        <h3>Upload Photos</h3>
        <div class="drop-zone" id="dropZone">
            <div class="icon">📁</div>
            <p><strong>Drag & drop images here</strong> or click to browse</p>
            <p style="margin-top:6px; font-size:0.85em;">PNG, JPG, GIF, WEBP · up to 32 MB per file · multiple files supported</p>
        </div>
        <div class="upload-progress" id="uploadProgress">
            <div class="progress-bar-wrap">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <p class="progress-text" id="progressText">Uploading and generating captions…</p>
        </div>
    </div>

    <!-- Image grid -->
    <div id="photoCount" style="margin-bottom:16px; color:#718096; font-size:.9em;"></div>
    <div class="image-grid" id="imageGrid">
        <div class="empty-state">
            <div class="icon">📷</div>
            <p>No photos yet. Upload some to get started!</p>
        </div>
    </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox">
    <button class="lightbox-close" onclick="closeLightbox()">✕</button>
    <button class="lightbox-nav" id="lightboxPrev" onclick="lightboxNav(-1)">‹</button>
    <img id="lightboxImg" src="" alt="">
    <div class="lightbox-caption" id="lightboxCaption"></div>
    <button class="lightbox-nav" id="lightboxNext" onclick="lightboxNav(1)">›</button>
</div>

<div class="toast" id="toast"></div>

<script>
const ALBUM_ID = {{ album_id | tojson }};
let albumData = null;
let lightboxIndex = 0;

async function loadAlbum() {
    const res = await fetch('/api/albums/' + ALBUM_ID);
    if (!res.ok) { document.getElementById('albumTitle').textContent = 'Album not found'; return; }
    albumData = await res.json();
    document.title = 'PhotoGallery — ' + albumData.name;
    document.getElementById('albumTitle').textContent = albumData.name;
    document.getElementById('albumMeta').textContent =
        (albumData.description || '') + (albumData.description ? ' · ' : '') +
        albumData.images.length + ' photo' + (albumData.images.length !== 1 ? 's' : '');
    renderImages();
}

function renderImages() {
    const grid = document.getElementById('imageGrid');
    const countEl = document.getElementById('photoCount');
    const imgs = albumData.images;
    countEl.textContent = imgs.length + ' photo' + (imgs.length !== 1 ? 's' : '');

    if (!imgs.length) {
        grid.innerHTML = `<div class="empty-state">
            <div class="icon">📷</div>
            <p>No photos yet. Upload some to get started!</p>
        </div>`;
        return;
    }

    grid.innerHTML = imgs.map((img, i) => `
        <div class="image-card" data-index="${i}">
            <img class="image-thumb"
                 src="/uploads/${ALBUM_ID}/${img.filename}"
                 alt="${escHtml(img.original_name)}"
                 onclick="openLightbox(${i})"
                 loading="lazy">
            <div class="image-body">
                <div class="image-caption" id="cap-${i}">
                    ${img.caption
                        ? escHtml(img.caption)
                        : '<span class="caption-loading">No caption yet</span>'}
                </div>
            </div>
            <div class="image-footer">
                <span>${formatDate(img.uploaded_at)}</span>
                <div style="display:flex;gap:6px;">
                    <button class="btn btn-sm btn-outline" onclick="regenerateCaption('${img.filename}', ${i})">↻ Caption</button>
                    <button class="btn btn-sm btn-danger" onclick="deleteImage('${img.filename}')">✕</button>
                </div>
            </div>
        </div>
    `).join('');
}

// --- Upload ---
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', e => {
    if (e.target.files.length) uploadFiles(e.target.files);
    fileInput.value = '';
});

async function uploadFiles(files) {
    const formData = new FormData();
    for (const f of files) formData.append('files', f);

    const prog = document.getElementById('uploadProgress');
    const bar = document.getElementById('progressBar');
    const text = document.getElementById('progressText');
    prog.style.display = 'block';
    bar.style.width = '30%';
    text.textContent = `Uploading ${files.length} photo${files.length > 1 ? 's' : ''} and generating captions…`;

    try {
        const res = await fetch('/api/albums/' + ALBUM_ID + '/upload', { method: 'POST', body: formData });
        bar.style.width = '100%';
        const data = await res.json();
        if (data.added && data.added.length) {
            albumData.images.push(...data.added);
            renderImages();
            document.getElementById('albumMeta').textContent =
                (albumData.description || '') + (albumData.description ? ' · ' : '') +
                albumData.images.length + ' photo' + (albumData.images.length !== 1 ? 's' : '');
            showToast(`${data.added.length} photo${data.added.length > 1 ? 's' : ''} uploaded with captions!`, 'success');
        }
        if (data.errors && data.errors.length) {
            showToast(data.errors.join(', '), 'error');
        }
    } catch (e) {
        showToast('Upload failed: ' + e.message, 'error');
    } finally {
        setTimeout(() => { prog.style.display = 'none'; bar.style.width = '0%'; }, 1200);
    }
}

async function regenerateCaption(filename, index) {
    const el = document.getElementById('cap-' + index);
    el.innerHTML = '<span class="caption-loading">Generating caption…</span>';
    try {
        const res = await fetch(`/api/albums/${ALBUM_ID}/images/${filename}/caption`, { method: 'POST' });
        const data = await res.json();
        if (data.caption) {
            albumData.images[index].caption = data.caption;
            el.textContent = data.caption;
            // Also update lightbox if open on this image
            if (document.getElementById('lightbox').classList.contains('open') && lightboxIndex === index) {
                document.getElementById('lightboxCaption').textContent = data.caption;
            }
            showToast('Caption updated!', 'success');
        } else {
            el.textContent = 'No caption generated';
        }
    } catch (e) {
        el.textContent = 'Caption error';
        showToast('Failed to generate caption', 'error');
    }
}

async function deleteImage(filename) {
    if (!confirm('Remove this photo from the album?')) return;
    const res = await fetch(`/api/albums/${ALBUM_ID}/images/${filename}`, { method: 'DELETE' });
    if (res.ok) {
        albumData.images = albumData.images.filter(i => i.filename !== filename);
        renderImages();
        showToast('Photo removed', 'success');
    } else showToast('Delete failed', 'error');
}

// --- Lightbox ---
function openLightbox(index) {
    lightboxIndex = index;
    updateLightbox();
    document.getElementById('lightbox').classList.add('open');
}

function closeLightbox() {
    document.getElementById('lightbox').classList.remove('open');
}

function lightboxNav(dir) {
    const imgs = albumData.images;
    lightboxIndex = (lightboxIndex + dir + imgs.length) % imgs.length;
    updateLightbox();
}

function updateLightbox() {
    const img = albumData.images[lightboxIndex];
    document.getElementById('lightboxImg').src = `/uploads/${ALBUM_ID}/${img.filename}`;
    document.getElementById('lightboxCaption').textContent = img.caption || 'No caption available';
    document.getElementById('lightboxPrev').style.display = albumData.images.length > 1 ? '' : 'none';
    document.getElementById('lightboxNext').style.display = albumData.images.length > 1 ? '' : 'none';
}

document.getElementById('lightbox').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeLightbox();
});
document.addEventListener('keydown', e => {
    if (!document.getElementById('lightbox').classList.contains('open')) return;
    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowLeft') lightboxNav(-1);
    if (e.key === 'ArrowRight') lightboxNav(1);
});

// --- Helpers ---
function escHtml(str) {
    return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function formatDate(iso) {
    return new Date(iso).toLocaleDateString(undefined, { year:'numeric', month:'short', day:'numeric' });
}
function showToast(msg, type='') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast show' + (type ? ' ' + type : '');
    setTimeout(() => t.className = 'toast', 3500);
}

loadAlbum();
</script>
</body>
</html>"""

NOT_FOUND_HTML = """<!DOCTYPE html>
<html><head><title>Not Found</title></head>
<body style="font-family:sans-serif;text-align:center;padding:80px;">
<h1>Album not found</h1>
<a href="/">← Back to Gallery</a>
</body></html>"""


if __name__ == '__main__':
    print("🚀 Starting PhotoGallery…")
    print(f"📦 Loading {MODEL_ID} (downloads ~1 GB on first run)…")
    load_model()
    print("✅ Model ready!")
    print("🌐 Open http://localhost:5001 in your browser")
    app.run(debug=False, host='0.0.0.0', port=5001)
