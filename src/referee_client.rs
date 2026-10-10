//! Synchronous JSON client for the GUI's dedicated referee child.
use crate::game::{Game, Piece};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc,
    time::Duration,
};
pub struct RefereeClient {
    child: Child,
    input: ChildStdin,
    replies: mpsc::Receiver<String>,
    pub state: Value,
    id: u64,
}
impl RefereeClient {
    pub fn spawn(log_directory: &std::path::Path) -> Result<Self, String> {
        let executable = std::env::var_os("SKALLAGRIM_REFEREE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::current_exe()
                    .unwrap()
                    .with_file_name(if cfg!(windows) {
                        "skallagrim-referee.exe"
                    } else {
                        "skallagrim-referee"
                    })
            });
        let mut command = Command::new(executable);
        command
            .arg("--log-dir")
            .arg(log_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("Cannot start dedicated referee: {e}"))?;
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (tx, replies) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut client = Self {
            child,
            input,
            replies,
            state: Value::Null,
            id: 0,
        };
        client.call(json!({"command":"snapshot"}))?;
        Ok(client)
    }
    pub fn call(&mut self, mut request: Value) -> Result<Value, String> {
        self.id += 1;
        request["id"] = json!(self.id);
        if request.get("revision").is_none()
            && self.state["revision"].is_u64()
            && !["snapshot", "new", "export", "pause", "validate"]
                .contains(&request["command"].as_str().unwrap_or(""))
        {
            request["revision"] = self.state["revision"].clone();
        }
        writeln!(self.input, "{request}")
            .and_then(|_| self.input.flush())
            .map_err(|e| e.to_string())?;
        let line = self
            .replies
            .recv_timeout(Duration::from_secs(5))
            .map_err(|e| format!("referee unavailable: {e}"))?;
        let response: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if response["id"].as_u64() != Some(self.id) {
            return Err("referee response mismatch".into());
        }
        self.state = response["state"].clone();
        if response["ok"] == true {
            Ok(response["result"].clone())
        } else {
            Err(response["error"]
                .as_str()
                .unwrap_or("referee rejected request")
                .into())
        }
    }
    pub fn game(&self) -> Result<Game, String> {
        let mut game = Game::default();
        let board: Vec<Piece> =
            serde_json::from_value(self.state["board"].clone()).map_err(|e| e.to_string())?;
        game.board = board
            .try_into()
            .map_err(|_| "invalid authoritative board")?;
        game.side =
            serde_json::from_value(self.state["side"].clone()).map_err(|e| e.to_string())?;
        game.outcome =
            serde_json::from_value(self.state["outcome"].clone()).map_err(|e| e.to_string())?;
        let history: Vec<std::sync::Arc<str>> =
            serde_json::from_value(self.state["history"].clone()).map_err(|e| e.to_string())?;
        game.history = history.into();
        game.last_move =
            serde_json::from_value(self.state["last_move"].clone()).map_err(|e| e.to_string())?;
        game.captures =
            serde_json::from_value(self.state["captures"].clone()).map_err(|e| e.to_string())?;
        Ok(game)
    }
    pub fn play(&mut self, mv: crate::game::Move) -> Result<Game, String> {
        self.call(json!({"command":"play","move":mv.to_string()}))?;
        self.game()
    }
}
impl Drop for RefereeClient {
    fn drop(&mut self) {
        let _ = writeln!(self.input, "{}", json!({"command":"quit"}));
        for _ in 0..20 {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
