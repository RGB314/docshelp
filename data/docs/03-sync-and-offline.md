# Sync and Offline Mode

## How sync works

Changes sync in real time over a WebSocket connection. When two people edit the same paragraph at the same time,
Nimbus merges the edits automatically; nothing is overwritten.

## Offline mode

Desktop and mobile apps keep a local copy of every notebook you have marked **Available offline**
(right-click a notebook → *Available offline*). Offline edits are queued and synced when you reconnect.
The web app does not support offline mode.

## Sync status

The cloud icon in the bottom-left corner shows sync status:

- Solid cloud: everything is synced.
- Cloud with arrows: sync in progress.
- Cloud with a red slash: sync is paused or failing. See *Troubleshooting* for fixes.
