# Records docs/demo/stream.gif from a real run: `vhs docs/demo/stream.tape` from the repo root.
# The echo model streams the answer in 4-character SSE chunks; the proxy rewrites the stream.
Output docs/demo/stream.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 640
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 18ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && python docs/demo/make_sample.py && S=docs/demo/.sample` Enter
Type `python -m llm_dlp_guard mock-llm --port 18083 --chunk 4 >/dev/null 2>&1 & python -m llm_dlp_guard serve --port 18082 --upstream http://127.0.0.1:18083 >/dev/null 2>&1 &` Enter
Type `until curl -s localhost:18082/ready >/dev/null; do sleep 0.5; done; clear` Enter
Sleep 4s
Show
Type `curl -sN localhost:18083/v1/chat/completions -d @$S/chat_stream.json | head -4   # raw upstream events` Enter
Sleep 3s
Type `curl -sN localhost:18082/v1/chat/completions -d @$S/chat_stream.json | jq -rj --unbuffered -R 'select(startswith("data: {")) | .[6:] | fromjson | .choices[0].delta.content // ""'; echo` Enter
Sleep 5s
Type `python -m llm_dlp_guard stream $S/answer.txt --chunk 4 --delay 0.004` Enter
Sleep 8s
Hide
Type `kill %1 %2` Enter
Sleep 1s
