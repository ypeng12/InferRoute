# 🚀 24/7 Free Cloud Deployment Guide

This guide details how to host both the **Static Technical Docs Hub** and the **Full Interactive Gateway (Playground & APIs)** online 24/7 completely for **free ($0 cost)**.

---

## 🎨 Part 1: Host Static Docs Hub on GitHub Pages (24/7 Free)

Since our Academic & Technical documentation hub (`docs/index.html`) is completely self-contained with client-side interactive cascade simulators and LaTeX math formulas, it runs perfectly on GitHub Pages:

### Step-by-Step GitHub Pages Setup:
1. **Push your code**: Ensure the latest commit containing the `docs/index.html` file is pushed to your GitHub repository (e.g. `github.com/ypeng12/InferRoute`).
2. **Open Repository Settings**: Go to your GitHub repository webpage and click the **Settings** tab.
3. **Navigate to Pages**: In the left sidebar under the "Code and automation" section, click **Pages**.
4. **Configure Build Source**:
   * Under **Build and deployment** ➔ **Source**, select **Deploy from a branch**.
   * Under **Branch**, select `main` (or your default branch) and change the folder dropdown from `/ (root)` to **`/docs`**.
5. **Save Configuration**: Click **Save**.
6. **Access Public URL**: GitHub will compile and launch the page in 1–2 minutes. Your docs hub will be permanently online at:
   `https://ypeng12.github.io/InferRoute/`

---

## 🧠 Part 2: Host Full Gateway Server on Hugging Face Spaces (24/7 Free)

Hugging Face Spaces allows you to host Dockerized Python web servers 24/7 for free. It automatically builds the application from our root `Dockerfile` and serves it securely.

### Step-by-Step Hugging Face Setup:
1. **Create Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. **Configure Space settings**:
   * **Space Name**: e.g., `InferRoute` or `inferroute-gateway`.
   * **License**: Choose `MIT` or open source.
   * **SDK**: Select **Docker** (very important).
   * **Docker Template**: Select **Blank** (do not select Gradio/Streamlit).
   * **Space Hardware**: Choose **CPU Basic (Free - 16GB RAM)**.
   * **Visibility**: Set to **Public**.
3. **Upload Files**:
   * Git clone the Space repository or upload the files directly using the Hugging Face web UI.
   * Upload the following files/directories to the Space repository:
     * `inferroute/` (complete folder)
     * `docs/` (complete folder)
     * `benchmarks/` (complete folder)
     * `requirements.txt`
     * `Dockerfile`
4. **Automatic Build**: Once files are uploaded, Hugging Face automatically detects the `Dockerfile`, compiles the dependencies, and deploys the gateway on port `7860`.
5. **Enjoy Live Playground**: Your gateway and interactive playground will be online 24/7 at:
   `https://huggingface.co/spaces/<your-username>/<your-space-name>`

---

## ⚡ Part 3: Alternative - Host Gateway on Render (Free Tier)

Render.com provides free container hosting with automatic GitHub integration:

1. **Connect GitHub**: Log in to [Render](https://render.com) and click **New** ➔ **Web Service**.
2. **Select Repository**: Connect your GitHub repository.
3. **Configure Environment**:
   * **Runtime**: Select **Docker**.
   * **Instance Type**: Select **Free**.
4. **Environment Variables**: Under the "Advanced" tab, add variables if you want to override default credentials (e.g. custom API keys).
5. **Deploy**: Render will build the container from the `Dockerfile` and expose it. 
   *(Note: Free tier Render instances go to sleep after 15 minutes of inactivity. Hugging Face Spaces is recommended for persistent 24/7 uptimes).*
