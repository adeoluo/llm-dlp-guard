# Records docs/demo/proxy.gif from a real run: `vhs docs/demo/proxy.tape` from the repo root.
# The upstream is the local echo model (`python -m llm_dlp_guard mock-llm`), so the demo shows
# exactly what a provider would receive and what the client gets back.
Output docs/demo/proxy.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 720
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 18ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && python docs/demo/make_sample.py && S=docs/demo/.sample` Enter
Type `python -m llm_dlp_guard mock-llm --port 18081 >/dev/null 2>&1 & python -m llm_dlp_guard serve --port 18080 --upstream http://127.0.0.1:18081 --model ../.bench-data/llm-dlp-guard/models/tagger.crf >/dev/null 2>&1 &` Enter
Type `until curl -s localhost:18080/ready >/dev/null; do sleep 0.5; done; clear` Enter
Sleep 6s
Show
Type `curl -s localhost:18080/v1/chat/completions -d @$S/chat.json | jq -r '.choices[0].message.content'` Enter
Sleep 3s
Type `curl -s localhost:18081/ | jq -r '.last_request.messages[0].content'   # what the provider saw` Enter
Sleep 4s
Type `curl -s localhost:18080/v1/chat/completions -d @$S/chat_key.json | jq -c .error` Enter
Sleep 3s
Type `curl -s localhost:18080/metrics | grep -E '^dlp_(findings|blocked)_total' | head -8` Enter
Sleep 5s
Hide
Type `kill %1 %2` Enter
Sleep 1s
