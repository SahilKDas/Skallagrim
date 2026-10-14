# Search response deadlines

The post-beta timing fix publishes completed search results before freeing the MCTS arena. The CLI records the deadline when it reads `go`, before joining a previous worker, copying models or starting threads. The GUI records it before arming the referee clock. Both use the same supervised search path.

Search computation observes the caller's absolute cutoff and leaves a short publication margin. A separate sleeping supervisor receives the completed move and full policy. If computation misses the response deadline, it requests cancellation and publishes one legal fallback, with `info string deadline-fallback` in CLI output. Late completion never emits a second move. The supervisor then joins the computation worker; it does not accumulate detached search workers.

On Windows, only the brief response-publication phase runs with above-normal thread priority, to avoid waiting behind compute/arena cleanup on a pinned CPU. Computation stays at normal thread priority. The previous response-thread priority is restored before joining the worker. Process priority, CPU affinity, referee clocks and transport allowances are unchanged.

The normal deadline remains a 300-ms request plus the documented 25-ms transport allowance. This is not a hard real-time operating system: machine suspension or starvation of the engine or client can still delay observation. The historical 114-minute timeout did not record enough context to prove its exact cause; new soak reports capture the bot, position, operation and fallback counts.

Evidence is recorded separately in benchmarks/search-fix-deadlines.json and benchmarks/search-fix-soak.json. The original beta reports and user overrides remain historical records. These checks measure timing and process stability, not playing strength.

## October 11 verification

All 66 Rust and 35 Python tests pass, along with formatting, Clippy and Ruff F. The 300-search probe had no responses above 325 ms; Beta's maximum was 301.74 ms across 100 trials with zero fallbacks. Two watchdog fallbacks occurred among the other 200 trials.

The continuous soak completed 7,200.891 seconds, 62,580 legal plies, 5,215 cancellation/reset/audit batches and 783 renders with no failures. Its 30-ms searches used 439 explicit watchdog fallbacks (Iota 153, Beta 138, Rho 148). These are legal fallback moves, not completed search results. All owned processes exited and the installation-specific process inventory was empty. This run passes without a user override. See benchmarks/search-fix-verification.json for binary hashes and summarized evidence.
