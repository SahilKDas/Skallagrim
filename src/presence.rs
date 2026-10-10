//! GUI-only Discord IPC. Nonblocking transports keep Discord off the game loop.
use serde_json::{Value, json};
use std::{
    io::{self, Read, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

trait Pipe: Read + Write + Send {}
impl<T: Read + Write + Send> Pipe for T {}
const UPDATE_INTERVAL: Duration = Duration::from_secs(5);
const RETRY_INTERVAL: Duration = Duration::from_secs(5);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(2);

fn frame(op: u32, value: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(value).unwrap();
    [
        op.to_le_bytes().as_slice(),
        (body.len() as u32).to_le_bytes().as_slice(),
        &body,
    ]
    .concat()
}
#[cfg(windows)]
fn connect() -> io::Result<Box<dyn Pipe>> {
    use std::os::windows::io::AsRawHandle;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetNamedPipeHandleState(
            handle: *mut std::ffi::c_void,
            mode: *const u32,
            max: *const u32,
            timeout: *const u32,
        ) -> i32;
    }
    for index in 0..10 {
        if let Ok(file) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!(r"\\.\pipe\discord-ipc-{index}"))
        {
            // PIPE_NOWAIT prevents read/write calls from holding worker shutdown.
            let mode = 1u32;
            if unsafe {
                SetNamedPipeHandleState(
                    file.as_raw_handle(),
                    &mode,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            } != 0
            {
                return Ok(Box::new(file));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotConnected,
        "Discord IPC unavailable",
    ))
}
#[cfg(unix)]
fn connect() -> io::Result<Box<dyn Pipe>> {
    use std::os::unix::net::UnixStream;
    for base in [
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os("TMPDIR"),
        Some("/tmp".into()),
    ]
    .into_iter()
    .flatten()
    {
        for index in 0..10 {
            if let Ok(stream) = UnixStream::connect(
                std::path::PathBuf::from(&base).join(format!("discord-ipc-{index}")),
            ) {
                stream.set_nonblocking(true)?;
                return Ok(Box::new(stream));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotConnected,
        "Discord IPC unavailable",
    ))
}
#[cfg(not(any(windows, unix)))]
fn connect() -> io::Result<Box<dyn Pipe>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Discord IPC unsupported",
    ))
}

fn pending(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock || error.raw_os_error() == Some(232) // ERROR_NO_DATA
}
struct Session {
    pipe: Box<dyn Pipe>,
    input: Vec<u8>,
    output: Vec<u8>,
    ready: bool,
    connected: Instant,
}
impl Session {
    fn new(pipe: Box<dyn Pipe>, id: &str, now: Instant) -> Self {
        Self {
            pipe,
            input: vec![],
            output: frame(0, &json!({"v":1,"client_id":id})),
            ready: false,
            connected: now,
        }
    }
    fn pump(&mut self, now: Instant) -> io::Result<()> {
        if !self.output.is_empty() {
            match self.pipe.write(&self.output) {
                Ok(0) => return Err(io::ErrorKind::BrokenPipe.into()),
                Ok(n) => {
                    self.output.drain(..n);
                }
                Err(e) if pending(&e) => {}
                Err(e) => return Err(e),
            }
        }
        let mut buffer = [0u8; 4096];
        match self.pipe.read(&mut buffer) {
            Ok(0) => return Err(io::ErrorKind::BrokenPipe.into()),
            Ok(n) => self.input.extend_from_slice(&buffer[..n]),
            Err(e) if pending(&e) => {}
            Err(e) => return Err(e),
        }
        if self.input.len() > 65536 {
            return Err(io::ErrorKind::InvalidData.into());
        }
        while self.input.len() >= 8 {
            let op = u32::from_le_bytes(self.input[..4].try_into().unwrap());
            let len = u32::from_le_bytes(self.input[4..8].try_into().unwrap()) as usize;
            if len > 65528 {
                return Err(io::ErrorKind::InvalidData.into());
            }
            if self.input.len() < 8 + len {
                break;
            }
            let data: Value = serde_json::from_slice(&self.input[8..8 + len])
                .map_err(|_| io::ErrorKind::InvalidData)?;
            self.input.drain(..8 + len);
            match op {
                1 if data["evt"] == "READY" => self.ready = true,
                1 if data["evt"] == "ERROR" => return Err(io::ErrorKind::ConnectionAborted.into()),
                2 => return Err(io::ErrorKind::ConnectionAborted.into()),
                3 => self.output.extend(frame(4, &data)),
                _ => {}
            }
        }
        if !self.ready && now.duration_since(self.connected) >= HANDSHAKE_TIMEOUT {
            return Err(io::ErrorKind::TimedOut.into());
        }
        Ok(())
    }
}
struct State {
    activity: Option<Value>,
    sent: Option<Value>,
    last: Option<Instant>,
    nonce: u64,
}
impl State {
    fn send(&mut self, session: &mut Session, now: Instant) {
        if session.ready
            && session.output.is_empty()
            && self.activity != self.sent
            && self
                .last
                .is_none_or(|t| now.duration_since(t) >= UPDATE_INTERVAL)
        {
            self.nonce += 1;
            session.output = frame(
                1,
                &json!({"cmd":"SET_ACTIVITY","args":{"pid":std::process::id(),"activity":self.activity},"nonce":self.nonce.to_string()}),
            );
            self.sent = self.activity.clone();
            self.last = Some(now);
        }
    }
}

fn run_worker(
    rx: mpsc::Receiver<Option<Value>>,
    id: String,
    mut connector: impl FnMut() -> io::Result<Box<dyn Pipe>>,
) {
    let mut session: Option<Session> = None;
    let mut state = State {
        activity: None,
        sent: None,
        last: None,
        nonce: 0,
    };
    let mut retry = Instant::now();
    let mut attempts = 0;
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Some(activity)) => state.activity = Some(activity),
            Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                if let Some(mut s) = session
                    && s.ready
                {
                    s.output = frame(
                        1,
                        &json!({"cmd":"SET_ACTIVITY","args":{"pid":std::process::id(),"activity":null},"nonce":"shutdown"}),
                    );
                    // One bounded best-effort clear; closing IPC also clears activity.
                    let _ = s.pipe.write(&s.output);
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        let now = Instant::now();
        if session.is_none() && now >= retry {
            attempts += 1;
            session = connector().ok().map(|p| Session::new(p, &id, now));
            retry = now
                + if attempts >= 12 {
                    Duration::from_secs(60)
                } else {
                    RETRY_INTERVAL
                };
            if session.is_some() {
                state.sent = None;
                state.last = None;
            }
        }
        if let Some(s) = session.as_mut() {
            if s.pump(now).is_err() {
                session = None;
                continue;
            }
            if s.ready {
                attempts = 0;
            }
            state.send(s, now);
        }
    }
}

pub struct Presence {
    tx: Option<mpsc::SyncSender<Option<Value>>>,
    worker: Option<thread::JoinHandle<()>>,
    start: u64,
    pub hint: String,
}
impl Presence {
    pub fn start(enabled: bool, id: Option<String>) -> Self {
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let valid = id
            .as_ref()
            .is_some_and(|s| (17..=20).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_digit()));
        if !enabled || !valid {
            return Self {
                tx: None,
                worker: None,
                start,
                hint: if !enabled {
                    "Discord presence disabled"
                } else {
                    "Discord: set --discord-app-id (or SKALLAGRIM_DISCORD_APP_ID)"
                }
                .into(),
            };
        }
        let id = id.unwrap();
        let (tx, rx) = mpsc::sync_channel(8);
        let worker = thread::spawn(move || run_worker(rx, id, connect));
        Self {
            tx: Some(tx),
            worker: Some(worker),
            start,
            hint: "Discord presence enabled (connects when Discord is running)".into(),
        }
    }
    pub fn update(&self, mode: &str, bots: &str, side: &str, result: Option<&str>) {
        if let Some(tx) = &self.tx {
            let activity = json!({"details":format!("{mode} · {bots}"),"state":result.map(|s|format!("Finished: {s}")).unwrap_or_else(||format!("{side} to move")),"timestamps":{"start":self.start}});
            let _ = tx.try_send(Some(activity));
        }
    }
}
impl Drop for Presence {
    fn drop(&mut self) {
        // Drop the sender if the bounded queue is full: disconnect also exits.
        if let Some(tx) = self.tx.take() {
            let _ = tx.try_send(None);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Mock {
        input: std::io::Cursor<Vec<u8>>,
        writes: Vec<u8>,
    }
    impl Read for Mock {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            let n = self.input.read(b)?;
            if n == 0 {
                Err(io::ErrorKind::WouldBlock.into())
            } else {
                Ok(n)
            }
        }
    }
    impl Write for Mock {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.writes.extend(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn session(now: Instant) -> Session {
        Session::new(
            Box::new(Mock {
                input: std::io::Cursor::new(frame(1, &json!({"evt":"READY"}))),
                writes: vec![],
            }),
            "123456789012345678",
            now,
        )
    }
    #[test]
    fn handshake_and_debounce() {
        let now = Instant::now();
        let mut s = session(now);
        s.pump(now).unwrap();
        assert!(s.ready);
        let mut state = State {
            activity: Some(json!({"state":"Attacker"})),
            sent: None,
            last: None,
            nonce: 0,
        };
        state.send(&mut s, now);
        assert_eq!(state.nonce, 1);
        s.pump(now).unwrap();
        state.activity = Some(json!({"state":"Defender"}));
        state.send(&mut s, now + Duration::from_secs(1));
        assert_eq!(state.nonce, 1);
        state.send(&mut s, now + UPDATE_INTERVAL);
        assert_eq!(state.nonce, 2);
    }
    #[test]
    fn missing_configuration_and_shutdown() {
        let p = Presence::start(true, None);
        assert!(p.tx.is_none());
        assert!(p.hint.contains("app-id"));
        drop(p);
    }
    #[test]
    fn reconnect_republishes() {
        let now = Instant::now();
        let mut s = session(now);
        s.pump(now).unwrap();
        let mut state = State {
            activity: Some(json!({"state":"Attacker"})),
            sent: None,
            last: None,
            nonce: 0,
        };
        state.send(&mut s, now);
        let mut replacement = session(now);
        replacement.pump(now).unwrap();
        state.sent = None;
        state.last = None;
        state.send(&mut replacement, now);
        assert_eq!(state.nonce, 2);
    }
    #[test]
    fn handshake_deadline() {
        let now = Instant::now();
        let mut s = Session::new(
            Box::new(Mock {
                input: std::io::Cursor::new(vec![]),
                writes: vec![],
            }),
            "123456789012345678",
            now,
        );
        assert!(s.pump(now + HANDSHAKE_TIMEOUT).is_err());
    }
    #[test]
    fn worker_shutdown_clears_mock_activity() {
        use std::sync::{Arc, Mutex};
        struct Shared {
            input: std::io::Cursor<Vec<u8>>,
            output: Arc<Mutex<Vec<u8>>>,
        }
        impl Read for Shared {
            fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
                let n = self.input.read(b)?;
                if n == 0 {
                    Err(io::ErrorKind::WouldBlock.into())
                } else {
                    Ok(n)
                }
            }
        }
        impl Write for Shared {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                self.output.lock().unwrap().extend(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let output = Arc::new(Mutex::new(Vec::new()));
        let writes = output.clone();
        let (tx, rx) = mpsc::sync_channel(8);
        let handle = thread::spawn(move || {
            run_worker(rx, "123456789012345678".into(), || {
                Ok(Box::new(Shared {
                    input: std::io::Cursor::new(frame(1, &json!({"evt":"READY"}))),
                    output: writes.clone(),
                }))
            })
        });
        tx.send(Some(json!({"state":"Attacker"}))).unwrap();
        thread::sleep(Duration::from_millis(160));
        tx.send(None).unwrap();
        handle.join().unwrap();
        let bytes = output.lock().unwrap();
        let mut offset = 0;
        let mut last = Value::Null;
        while offset + 8 <= bytes.len() {
            let len =
                u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
            last = serde_json::from_slice(&bytes[offset + 8..offset + 8 + len]).unwrap();
            offset += 8 + len;
        }
        assert_eq!(last["cmd"], "SET_ACTIVITY");
        assert!(last["args"]["activity"].is_null());
    }
}
