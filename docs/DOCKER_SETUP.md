# Docker on Windows 10 - step by step

> **A manual procedure, not a script.** Installing WSL and Docker changes Windows settings, needs an
> administrator PowerShell and a reboot, so every command here is meant to be run by hand, one at a time.
> Status 2026-10-04: the image is built and tested **in CI only** (first green run 2026-10-04: image
> 1.96 GB, smoke test 3/3, 153/153 self-test cases inside the image; `DECISIONS.md` C17, C18). It has
> not been built on the development PC yet, so the steps below are not yet tried end to end.

## 1. What Docker is

**Docker puts an app in a sealed box that runs the same anywhere.** The box holds a small Linux,
Python, every package, our code, the data, the embedding model and the built search index. Whoever
has Docker can run the box with one command - no venv, no `pip install`, no "works on my machine".

Three words you will hear all the time:

| Word | What it is |
|---|---|
| **Dockerfile** | the recipe: a list of steps that builds the box |
| **image** | the finished box, stored on disk, never changes |
| **container** | one running copy of an image |

A container is NOT a full virtual machine: it shares the Linux kernel that Docker Desktop runs in
the background (through WSL 2), so it starts in a second and uses little memory.

Why bother for this project: (1) a reviewer runs the whole system with one command, and (2) the same
image runs on a laptop, in CI and on AWS.

## 2. Before you start

Docker images are large, so these steps keep them off the system drive. The paths below use `D:`
as the second drive; use any drive with a few GB free. Open PowerShell and check the free space:

```powershell
Get-PSDrive C, D        # "Free" column. Write down C:'s number to compare after the install.
```

Expect about 1-1.5 GB to land on C: anyway (the WSL program, Docker settings and logs - an estimate
from the research, not measured). Everything big goes to D: if you follow the steps below.

Requirements: Windows 10 22H2 (build 19045, Docker Desktop's minimum) or newer, with hardware
virtualization turned on (Task Manager > Performance > CPU shows "Virtualization: Enabled").

## 3. Install WSL 2 (Windows Subsystem for Linux)

Docker Desktop runs its Linux inside WSL 2, so WSL comes first.

1. Start menu -> type `PowerShell` -> right-click **Windows PowerShell** -> **Run as administrator**.
2. Run:
   ```powershell
   wsl --install --no-distribution
   ```
   - `wsl --install` turns on the Windows features WSL needs and installs the WSL program.
   - `--no-distribution` = do NOT also install Ubuntu. Plain `wsl --install` would add an Ubuntu
     Linux on C: (several GB). Docker Desktop brings its own small Linux, so we don't need Ubuntu.
3. **Reboot** when it asks (it may not ask if "Virtual Machine Platform" was already on - reboot anyway).
4. After the reboot, in a normal PowerShell:
   ```powershell
   wsl --version           # must show "WSL version: 2.1.5" or higher
   ```
   If it is older: `wsl --update` (admin PowerShell).

### Move WSL's swap file off C: (recommended)

WSL keeps a swap file (extra "memory" on disk) on C: by default. Tell it to use D::

1. Make the folder: `New-Item -ItemType Directory -Force D:\WSL`
2. Open Notepad, paste the three lines below, and save as `.wslconfig` in your user folder
   (`%UserProfile%\.wslconfig`, e.g. `C:\Users\<you>\.wslconfig`)
   (in the Save dialog choose "All files (\*.\*)" so Notepad does not add `.txt`):
   ```ini
   [wsl2]
   memory=6GB
   swapFile=D:\\WSL\\swap.vhdx
   ```
   - `memory=6GB` caps how much RAM the Linux side may take (the build needs ~2-3 GB).
   - `swapFile=` the double backslashes are required in this file.
3. Apply it: `wsl --shutdown` (it restarts by itself next time it is needed).

## 4. Install Docker Desktop with the program AND its data on D:

1. Download "Docker Desktop for Windows - x86_64" from
   https://docs.docker.com/desktop/setup/install/windows-install/ (about 640 MB).
   **Save it to `D:\Installers\`** (create the folder; in the browser use "Save as", or move it
   out of Downloads afterwards - Downloads is on C:).
2. Admin PowerShell (as in step 3.1), then:
   ```powershell
   Start-Process -Wait -FilePath "D:\Installers\Docker Desktop Installer.exe" -ArgumentList @(
     'install','--quiet','--accept-license','--backend=wsl-2',
     '--installation-dir=D:\Docker\DockerDesktop',
     '--wsl-default-data-root=D:\DockerData\wsl')
   ```
   What each part does:
   - `Start-Process -Wait` - run the installer and wait until it is finished.
   - `install` - install (not uninstall/update).
   - `--quiet` - no installer windows; it just runs. It takes a few minutes and prints nothing.
   - `--accept-license` - you accept the Docker Subscription Service Agreement now instead of
     clicking it on first start. Docker Desktop is free for personal use, education and small
     businesses (under 250 employees AND under $10 million revenue), so this project is free.
   - `--backend=wsl-2` - run the Linux part on WSL 2 (not the older Hyper-V way).
   - `--installation-dir=D:\Docker\DockerDesktop` - the program goes to D: (default: C:\Program Files).
   - `--wsl-default-data-root=D:\DockerData\wsl` - **all images and containers** go to D:. This is
     the big one: it grows by several GB.
   - Keep the data folder OUTSIDE the program folder (as above): an open Docker bug (Sept 2026, on
     Windows 10) deleted the program folder during an update - with everything inside it.
3. Start **Docker Desktop** from the Start menu. Wait until the bottom-left corner says
   **"Engine running"** (green). Skip the sign-in; an account is not needed.
4. Check where the data went: **Settings (gear icon) > Resources > Advanced > Disk image location**
   must show `D:\DockerData\wsl`. If it shows a C: path, change it there to `D:\DockerData\wsl`,
   then **Apply & restart** (Docker moves the disk for you).
5. Check it works - in a NEW normal PowerShell window (a new window picks up the new `docker` command):
   ```powershell
   docker --version                    # prints the Docker version
   docker run --rm hello-world         # downloads a tiny test image, prints "Hello from Docker!"
   Get-ChildItem D:\DockerData\wsl -Recurse -Filter *.vhdx    # the data disk files are on D:
   Get-PSDrive C                       # compare C:'s free space with step 2
   ```
   - `docker run` = start a container from an image (downloading the image first if needed).
   - `--rm` = delete the container when it stops (the image stays).
   - Do not use `docker info`'s "Root Dir" to check the location: that is a path INSIDE the Linux VM.

> **Windows 10 note.** Docker supports Windows versions still inside Microsoft's support timeline.
> Reports from September 2026 show Docker Desktop 4.90 running on Windows 10 22H2, but a future
> Docker release could drop Windows 10. If an update refuses to install, stay on the version you have.

## 5. Build the image

```powershell
cd D:\rag-support-agent
docker build -t rag-support-agent .
```
- `docker build` - follow the `Dockerfile` recipe and produce an image.
- `-t rag-support-agent` - the image's name ("tag"), so you can refer to it later.
- `.` (the dot!) - "the build context is this folder": Docker sends this folder (minus everything in
  `.dockerignore`, e.g. `.venv`, `.env`, `data/raw`) to the builder. Easy to forget the dot.

What you will see: one block per Dockerfile step, like `[3/14] RUN pip install torch==2.14.1 ...`
(Docker counts only FROM, WORKDIR, RUN and COPY lines, so its numbers differ from the numbered
comments in the `Dockerfile`; "steps 8-11" below means those comments).
The first build downloads Python, CPU PyTorch (~200 MB), the other packages and the embedding model
(~130 MB), then builds the search index (~1.5 min). Expect roughly 10-20 minutes the first time
(estimate - depends on the internet). A second build only re-runs the steps from the first one
whose files changed: edit a file in `ragagent/` and steps 8-11 re-run (including the ~1.5 min index
build, but no package or model download); edit only `evals/` and just step 11 re-runs; edit
`requirements.txt` and the package install re-runs too.

```powershell
docker images                      # rag-support-agent should be listed, with its SIZE
```
Expected size: about 1.96 GB (measured in the first CI build, DECISIONS.md C17). Most of it is
PyTorch, the rest the other packages + the model.

## 6. Run it

```powershell
docker run --env-file .env -p 8000:8000 rag-support-agent
```
- `--env-file .env` - hands your `.env` lines (`GEMINI_API_KEY=...`, `LLM_PROVIDER=...`) to the
  container as environment variables **when it starts**. The key is never inside the image, so the
  image is safe to share; whoever runs it brings their own key.
- `-p 8000:8000` - "publish" a port: PC port 8000 -> container port 8000. Without it the API
  runs, but nothing outside the container can reach it.
- `rag-support-agent` - the image to run.

Then open **http://localhost:8000/docs** - the same FastAPI page as with the venv. Try:
1. `GET /health` -> `{"status": "ok", "index_built": true, ...}`
2. `GET /documents` -> the 15 documents.
3. `POST /ask` with `{"question": "What late charge applies if a loan payment is late?"}`
   (1 Gemini call) - a cited answer, or a refusal.

Or check everything at once, from a second PowerShell window (the same check CI runs, no LLM call):
```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Stop it: **Ctrl+C** in the window where it runs.

Useful variants:
```powershell
docker run --rm --env-file .env -p 8000:8000 rag-support-agent                 # --rm: delete the container when stopped
docker run -d --name rag --env-file .env -p 8000:8000 rag-support-agent        # -d: run in the background, named "rag"
docker run -d --name rag --env-file .env -p 8000:8000 -v rag-llm-cache:/app/.cache/llm rag-support-agent
#   -v name:/path = a "volume": a folder that survives the container. Here it keeps the LLM answer
#   cache, so a repeated question costs no quota even after you delete and restart the container.
docker run --rm --network none rag-support-agent python -m ragagent.agent.loop --selftest
#   run any command inside the image instead of the API: here the agent loop's self-test, with no network at all
```

## 7. Everyday commands

`rag` below = a container's name: the one you gave with `--name rag`, or the random name Docker
picked (NAMES column of `docker ps`).

| Command | What it does |
|---|---|
| `docker ps` | running containers (with "healthy" / "unhealthy" from the Dockerfile's HEALTHCHECK) |
| `docker ps -a` | also the stopped ones |
| `docker logs rag` | everything the container printed (`-f` = keep following, Ctrl+C to stop) |
| `docker stop rag` | stop the container named `rag` (`docker start rag` starts it again) |
| `docker rm rag` | delete a stopped container (the image stays) |
| `docker exec -it rag bash` | open a shell INSIDE the running container (`exit` to leave) |
| `docker images` | the images on disk and their size |
| `docker image rm rag-support-agent` | delete an image |
| `docker system df` | how much disk Docker uses, by kind |
| `docker builder prune` | delete the build cache (next build is slower, frees space) |
| `docker system prune` | delete stopped containers, unused networks, dangling images and build cache (asks first) |

The data disk on D: grows and does not shrink by itself; the two `prune` commands free space inside it.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `wsl --install`: "requires elevation" | Use an **administrator** PowerShell (step 3.1). |
| Docker Desktop: "WSL needs updating" / "WSL 2 installation is incomplete" | Admin PowerShell: `wsl --update`, then restart Docker Desktop. |
| `docker : The term 'docker' is not recognized` | Close and reopen PowerShell (the install changed PATH). Still missing: is Docker Desktop installed and started? |
| `error during connect` / `Cannot connect to the Docker daemon` | Docker Desktop is not running. Start it and wait for "Engine running". |
| Build: `toomanyrequests` while pulling `python:3.13-slim` | Docker Hub limits anonymous downloads (100 per 6 hours per IP). Wait, or create a free Docker Hub account and `docker login`. |
| Build fails in a `pip install` step with a network error | Run the same `docker build` again: finished steps are cached, it continues where it failed. |
| Build fails in step 10 (ingest) with an error about offline mode / files not found in the cache | `EMBED_MODEL` in the Dockerfile and `MODEL_NAME` in `ragagent/rag/embed.py` differ. Make them equal. |
| Run: `port is already allocated` | Something already uses port 8000 (your venv `uvicorn`, or an older container: `docker ps`). Stop it, or use `-p 8001:8000` and open http://localhost:8001/docs. |
| `/ask` or `/agent` answers **503 "LLM unavailable"** | The key did not arrive. "no Gemini API key" in the message = no key at all: did you forget `--env-file .env`? Otherwise, in `.env`: `GEMINI_API_KEY=abc...` with **no quotes, no spaces around `=`, no comment after the value** - Docker copies the line literally (python-dotenv is more forgiving, so it can work in the venv and fail in Docker). Restart the container after editing `.env`. If the message says "daily free quota", wait until tomorrow. |
| `LLM_PROVIDER=ollama` works in the venv but not in Docker | Inside a container, `localhost` is the container itself, not your PC. Add `OLLAMA_HOST=http://host.docker.internal:11434` to `.env` (the Ollama client reads it). |
| You changed code or documents but the container behaves the old way | An image is a snapshot. Run `docker build -t rag-support-agent .` again, then a new `docker run`. |
| `docker ps` shows **unhealthy** | `docker logs rag` shows why the API is not answering `/health`. |
| D: is filling up | `docker system df`, then `docker builder prune` / `docker system prune`. |

## 9. What was checked without Docker (2026-10-01)

Nothing here proves the image builds - only that its parts are consistent:
- every package in `requirements.txt` + CPU torch 2.14.1 resolves to a ready-made Linux wheel for
  Python 3.13 (`pip install --dry-run --platform manylinux... --python-version 3.13 --only-binary=:all:`), so the slim image needs no compiler;
- a script read the Dockerfile against the repo: every `COPY` source exists and is not in
  `.dockerignore`, `ragagent.api:app` is the FastAPI app, `EMBED_MODEL` = `embed.MODEL_NAME`,
  `HF_HOME` = the folder `config.py` uses, no key in the image;
- the embedding model loads with `HF_HUB_OFFLINE=1` (as in the image);
- `scripts/smoke_test.py` passes against the venv's `uvicorn`, and all 14 self-test modules the CI
  runs pass in the venv;
- review run (2026-10-01): the exact files the image would get (`.dockerignore` + the `COPY` lines)
  copied to a folder on D:, then run there with the venv's Python, NO `.env`, `HF_HUB_OFFLINE=1` and
  the network blocked by a dead proxy: `ragagent.rag.ingest` built 1,860 chunks from 15 documents,
  the CMD (`uvicorn ragagent.api:app`) started, `smoke_test.py` passed 3/3, `/ask` and `/agent`
  without a key answered 503 "no Gemini API key", and the CI self-test loop (copied from
  `docker.yml`) passed all 14 modules. Still not a Linux image: package wheels, `useradd` and
  file permissions are only checked by a real build.

The first real build was the green CI run of 2026-10-04 (run 37205328703); a local `docker build` +
`docker run` + `scripts\smoke_test.py` has not been done yet.

## 10. The image's design in brief

- **What is in the image?** Python 3.13 slim, CPU-only PyTorch, the packages, the code, the document
  text, the embedding model and the Chroma index - built at image build time, so the container
  answers immediately and needs no internet except for the LLM.
- **Why CPU-only torch?** The default Linux torch pulls the NVIDIA CUDA libraries, several GB we
  never use: embeddings for 15 documents run fine on a CPU.
- **Where is the API key?** Not in the image. It is passed at run time (`--env-file`); on AWS it
  would be an IAM role, no key at all.
- **Why that step order?** Docker caches each step; slow, stable steps (packages, model) first,
  code last, so a code change never re-downloads the ~1 GB of packages and the model.
- **Why a non-root user?** If the web server were ever compromised, the attacker would not be root
  inside the container.
- **What does CI check?** It builds the image on every push to `main` and every pull request, starts it without any key, calls
  `/health`, `/documents` and an off-topic `/ask` (refused before any LLM call), then runs every
  offline self-test inside the image with the network switched off.
