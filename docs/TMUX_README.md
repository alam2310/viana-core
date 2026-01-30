The Golden Rule: Nearly every command in tmux starts with the "Prefix" combination. You must press Ctrl + b, release both, and then press the command key.

1. Session Management (The most important part)
Start a new named session:

Bash
tmux new -s training_job
Detach (Leave running in background): Ctrl+b then d

List all running sessions:

Bash
tmux ls
Re-attach to a session:

Bash
tmux attach -t training_job
Kill a session (Stop training & close tmux): Ctrl+b then :kill-session (or just type exit in the terminal until it closes).

2. Window Management (Like browser tabs)
Create a new window: Ctrl+b then c

Switch to the next window: Ctrl+b then n

Switch to the previous window: Ctrl+b then p

Rename current window: Ctrl+b then ,

3. Pane Management (Split screen)
Split vertically (Left/Right): Ctrl+b then %

Split horizontally (Top/Bottom): Ctrl+b then "

Switch focus between panes: Ctrl+b then Arrow Keys

Close the current pane: Type exit or Ctrl+d

4. Scrolling (Crucial for reading logs)
By default, you cannot scroll up with your mouse wheel in tmux.

Enter Scroll Mode: Ctrl+b then [ (Now you can use PageUp, PageDown, or Arrow keys to scroll through your logs).

Exit Scroll Mode: Press q

Recommended Workflow for You
tmux new -s viana_train

(Inside tmux) docker exec -it <container> bash

(Inside container) python3 process_video.py

Ctrl+b then d (Detach)

sudo systemctl isolate multi-user.target (Kill GUI to save RAM)

(Later) tmux attach -t viana_train (Check progress)