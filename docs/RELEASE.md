# Windows beta installation and qualification

Extract the entire archive into a writable directory, including paths containing spaces. Keep the three executables, models and documentation together. Launch `skallagrim-gui.exe`; Python and Rust are not required for desktop play. Python training needs the dependencies in python/requirements.txt.

No developer environment variables are necessary. If configured previously, remove SKALLAGRIM_ENGINE, SKALLAGRIM_REFEREE and SKALLAGRIM_DISCORD_APP_ID when checking clean installation. Use --no-discord to disable presence explicitly. Missing application ID disables presence normally.

Press J for referee evidence. PgUp/PgDn inspect certificate nodes and Home returns to the live board. Previewing a node never changes the authoritative position. D/F request human draw/fort decisions with a recorded reason. S saves an audited replay; --replay PATH loads it. See REFEREE.md for full controls and ruling semantics.

Release qualification records automated protocol, rendering, replay and soak checks separately from native mouse/keyboard testing. A process soak does not establish interactive GUI reliability or playing strength. General fort recognition is conservative; budgets or unsupported formations return unknown. The operational repetition policy is documented in RULES.md.

Manual checklist: N starts a new game, H cycles human roles, B cycles bots, U undoes a move, S saves. Check all three bots, resize, legal-move feedback, save/load, proof branch preview and return to live board, cancellation and window shutdown. Native controls are unavailable to the current automation; these checks must remain untested until a human or enabled native controller records them.

See benchmarks/release-validation.json for measured results. Full league training, live Discord and strength gauntlets are outside this release cleanup.

## Qualification outcome

The first two-hour soak attempt stopped after about 114 minutes on a search-response timeout (2.03-second watchdog). It completed 50,629 legal plies, 4,219 replay audits and 633 renders first. All owned processes exited and a separate process inventory found no leftovers. This is a failed stability gate, not a clean two-hour pass. A separately recorded recovery run does not erase it.

A 60-search timing probe returned all moves, but one Beta search exceeded the documented 325-ms allowance. The timeout cause remains unresolved. Native manual GUI interactions remain untested. The versioned archive is an experimental candidate, pending these qualification checks.
