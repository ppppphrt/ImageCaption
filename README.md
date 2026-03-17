# PhotoGallery — AI-Powered Personal Image Gallery

A web-based personal gallery that lets you organize photos into albums with automatic AI-generated captions using **Salesforce/blip-image-captioning-base** from HuggingFace.

---

## Features

- Create and manage multiple photo albums
- Upload multiple images at once (drag & drop supported)
- Automatic AI captions generated for every photo on upload
- Lightbox viewer with keyboard navigation (← → Esc)
- Regenerate captions on demand
- Delete individual photos or entire albums

---

## Requirements

- Python 3.8+
- ~1 GB disk space (for model weights, downloaded once)

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/ppppphrt/ImageCaption.git
cd ImageCaption
```

**2. Install dependencies**
```bash
pip install flask pillow torch transformers werkzeug python-dotenv
```

**3. Set up your environment file**

Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Or create it manually:
```
HF_API_TOKEN=your_huggingface_token_here
```

> Get a free token at https://huggingface.co/settings/tokens (Read access is enough).
> The token is optional but recommended to avoid rate limits when downloading the model.

---

## Running the App

```bash
python app.py
```

On **first run**, the model (`Salesforce/blip-image-captioning-base`, ~1 GB) will be downloaded automatically from HuggingFace and cached locally. Subsequent runs load from cache.

Once started, open your browser at:
```
http://localhost:5001
```

---

## Usage

1. **Create an album** — Click **"+ New Album"**, enter a name and optional description
2. **Upload photos** — Open an album, then drag & drop images or click **"Upload Photos"**
3. **View captions** — Captions are generated automatically on upload and shown below each photo
4. **Browse** — Click any photo to open the full-size lightbox viewer
5. **Regenerate caption** — Click the **↻ Caption** button on any photo to re-run AI captioning
6. **Delete** — Remove individual photos or entire albums with the delete buttons

---

## Project Structure

```
ImageCaption/
├── app.py          # Main Flask application
├── .env            # Your HuggingFace token (not committed)
├── .gitignore
├── README.md
├── uploads/        # Uploaded images, organized by album (auto-created)
└── data/
    └── albums.json # Album metadata and captions (auto-created)
```

---

## ML Component

This app uses **[Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)** — a vision-language model that generates natural language descriptions of images. The model runs locally via the `transformers` library, with weights sourced from HuggingFace.
