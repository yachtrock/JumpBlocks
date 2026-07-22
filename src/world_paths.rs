//! Platform-specific world save directory resolution.
//!
//! Worlds are stored in:
//!   macOS:   ~/Library/Application Support/JumpBlocks/worlds/
//!   Windows: C:\Users\<user>\AppData\Roaming\JumpBlocks\worlds\  (or Documents)
//!   Linux:   ~/.local/share/JumpBlocks/worlds/

use std::path::{Path, PathBuf};
use std::fs;

const APP_NAME: &str = "JumpBlocks";
const WORLDS_DIR: &str = "worlds";
const DEFAULT_WORLD: &str = "default";

/// Returns the base directory for all JumpBlocks worlds.
/// Creates the directory if it doesn't exist.
pub fn worlds_base_dir() -> PathBuf {
    let base = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
        .join(WORLDS_DIR);
    let _ = fs::create_dir_all(&base);
    base
}

/// Returns the directory for a specific world by name.
pub fn world_dir(name: &str) -> PathBuf {
    worlds_base_dir().join(name)
}

/// Returns the default world directory.
pub fn default_world_dir() -> PathBuf {
    world_dir(DEFAULT_WORLD)
}

/// List all existing world names.
pub fn list_worlds() -> Vec<String> {
    let base = worlds_base_dir();
    let Ok(entries) = fs::read_dir(&base) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| {
            let e = e.ok()?;
            if e.file_type().ok()?.is_dir() {
                Some(e.file_name().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect()
}

/// Check if a world exists on disk.
pub fn world_exists(name: &str) -> bool {
    world_dir(name).is_dir()
}

/// Delete a world from disk.
pub fn clear_world(name: &str) -> std::io::Result<()> {
    let dir = world_dir(name);
    if dir.is_dir() {
        fs::remove_dir_all(&dir)?;
    }
    Ok(())
}

/// Create/ensure a world directory exists.
pub fn ensure_world(name: &str) -> std::io::Result<PathBuf> {
    let dir = world_dir(name);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}
