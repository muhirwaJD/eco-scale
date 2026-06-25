# Deploy the Eco-Scale Console to Hugging Face Spaces

The console + API are served by one container (the root `Dockerfile`) on port
7860. Simulation mode works publicly; Live / Stage-2 tabs auto-disable (no cluster).

## One-time setup
1. Create a free account at https://huggingface.co and a write **access token**
   (Settings → Access Tokens → New token, role *write*).
2. Create a new Space: https://huggingface.co/new-space
   - **SDK:** Docker → *Blank*
   - **Space hardware:** CPU basic (free)
   - Name it e.g. `eco-scale-console`.

## Push the code
From the project root:

```bash
# 1. log in (paste your write token when asked)
pip install -U huggingface_hub
huggingface-cli login

# 2. add the Space as a git remote (replace <user>)
git remote add space https://huggingface.co/spaces/<user>/eco-scale-console

# 3. put the Space README (with the HF frontmatter) at the repo root for the push
cp deploy/huggingface/README.md HF_README.md   # keep your project README intact
#    -- OR simply let the Space use its own README created in the UI and set
#       app_port=7860 + sdk=docker there.

# 4. push the current branch to the Space
git push space main
```

Hugging Face will build the `Dockerfile` (React build + Python serve) and start it
on port 7860. First build takes ~10–15 min. When it's live, the Space URL is your
**deployed link** — paste it at the top of the main README.

## Verify
- Open the Space URL → the console loads in **Simulation** mode.
- Click **Play** → the agent vs HPA animates on a real trace.
- The **Live cluster** and **Real A/B** tabs are greyed out (no cluster on HF) —
  expected; demo those locally / in the video.

## Notes
- The Space serves everything on one origin, so the frontend calls the API at
  `/config`, `/sim/step`, … (built with `VITE_API_URL=""`).
- If the build runs out of memory, switch the Space hardware to a larger free CPU
  tier, or reduce to the dashboard-only deploy (Streamlit Cloud) as a fallback.
