#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(prefix: &str) -> Self {
        let id = uuid_like();
        let path = std::env::temp_dir().join(format!("af-tools-{prefix}-{id}"));
        std::fs::create_dir_all(&path).unwrap();
        // Canonicalise so comparisons with `pwd` output survive symlinked
        // temp dirs (macOS /var -> /private/var).
        TempDir(std::fs::canonicalize(&path).unwrap())
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "{}-{nanos}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

/// Whether a process with `pid` still exists (Unix).
#[cfg(unix)]
pub fn process_alive(pid: u32) -> bool {
    // On Linux a killed process whose parent is gone can linger as a zombie
    // until its new parent reaps it; a zombie runs nothing, so count it dead.
    if Path::new("/proc/self/stat").exists() {
        return match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat
                .rsplit_once(')')
                .map(|(_, rest)| !rest.trim_start().starts_with('Z'))
                .unwrap_or(false),
            Err(_) => false,
        };
    }
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Poll `cond` for up to five seconds.
pub async fn eventually(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if cond() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    cond()
}
