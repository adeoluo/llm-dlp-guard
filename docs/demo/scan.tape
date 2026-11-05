# Records docs/demo/scan.gif from a real run: `vhs docs/demo/scan.tape` from the repo root, after
# `make train` has written the tagger to ../.bench-data/llm-dlp-guard/models/tagger.crf.
Output docs/demo/scan.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 720
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 18ms

Hide
Type `export PS1='$ ' && source .venv/bin/activate && python docs/demo/make_sample.py && export M=../.bench-data/llm-dlp-guard/models/tagger.crf && clear` Enter
Sleep 2s
Show
Type `cat docs/demo/.sample/prompt.txt` Enter
Sleep 1.5s
Type `python -m llm_dlp_guard scan docs/demo/.sample/prompt.txt --model $M` Enter
Sleep 4s
Type `python -m llm_dlp_guard redact docs/demo/.sample/prompt.txt --model $M` Enter
Sleep 5s
