# Discord Rich Presence

The GUI publishes mode, selected bot, side to move, result and elapsed session time through local IPC, following the [official Discord RPC protocol](https://github.com/discord/discord-api-docs/blob/main/developers/topics/rpc.mdx). Positions, paths and human referee identities are excluded.

Create an application named **Skallagrim** in the [Discord developer portal](https://discord.com/developers/applications), then supply its Application ID:

```powershell
target/release/skallagrim-gui.exe --discord-app-id YOUR_APPLICATION_ID
# Alternatively:
$env:SKALLAGRIM_DISCORD_APP_ID="YOUR_APPLICATION_ID"
target/release/skallagrim-gui.exe
```

`--no-discord` disables presence. A missing/invalid ID disables IPC and shows a configuration hint at startup. `--render` never publishes presence. No ID is bundled; live publishing awaits configuration. Headless engines, referee and training tools do not publish presence.

The background worker uses nonblocking transport, a two-second handshake deadline, five-second update debounce, and reconnect backoff (five seconds initially, sixty seconds after twelve failed attempts). Shutdown clears activity when possible and closes IPC. Discord absence/rejection never interrupts gameplay. Mocked tests cover handshake, reconnect, debounce and shutdown; live validation requires an application ID.
