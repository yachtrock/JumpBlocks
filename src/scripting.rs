//! Rhai scripting integration for UI hot-reload.
//!
//! Runs `.rhai` scripts on the UI thread with access to canvas drawing,
//! FFD simulations, input state, and game data. Scripts are automatically
//! hot-reloaded when the source file changes on disk.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use bevy::input::keyboard::KeyCode;
use jumpblocks_ui::canvas::Canvas;
use jumpblocks_ui::ffd::FfdSim;
use rhai::{Array, Dynamic, Engine, Map, Scope, AST, FLOAT, INT};

use crate::{GameUiData, GameUiEvent};

// ---------------------------------------------------------------------------
// Draw command buffer
// ---------------------------------------------------------------------------

/// Draw operation recorded by the script, replayed on Canvas after execution.
enum DrawOp {
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    },
    RectFfd {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
        ffd_id: i64,
    },
    Text {
        x: f32,
        y: f32,
        text: String,
        font_size: f32,
        color: [f32; 4],
    },
    TextFfd {
        x: f32,
        y: f32,
        text: String,
        font_size: f32,
        color: [f32; 4],
        ffd_id: i64,
    },
    PushClip {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
    },
    PopClip,
}

// ---------------------------------------------------------------------------
// Shared state (accessed by Rhai-registered closures via Arc<Mutex>)
// ---------------------------------------------------------------------------

struct Shared {
    draw_ops: Vec<DrawOp>,
    ffd_sims: HashMap<i64, FfdSim>,
    next_ffd_id: i64,
    events: Vec<GameUiEvent>,
    keys_just_pressed: HashSet<String>,
    window_size: [f32; 2],
    game_data: Dynamic,
}

impl Shared {
    fn new() -> Self {
        Self {
            draw_ops: Vec::with_capacity(256),
            ffd_sims: HashMap::new(),
            next_ffd_id: 1,
            events: Vec::new(),
            keys_just_pressed: HashSet::new(),
            window_size: [0.0; 2],
            game_data: Dynamic::UNIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Script engine
// ---------------------------------------------------------------------------

/// Rhai script engine for hot-reloadable UI scripting.
pub struct ScriptEngine {
    engine: Engine,
    ast: Option<AST>,
    shared: Arc<Mutex<Shared>>,
    /// Persistent script state (Rhai Map). Survives hot reloads.
    state: Dynamic,
    script_path: PathBuf,
    last_modified: Option<SystemTime>,
    frame_count: u64,
}

impl ScriptEngine {
    pub fn new(script_path: impl Into<PathBuf>) -> Self {
        let shared = Arc::new(Mutex::new(Shared::new()));
        let engine = build_engine(Arc::clone(&shared));
        let script_path = script_path.into();

        let ast = load_script(&engine, &script_path);
        let last_modified = file_mtime(&script_path);

        // Try calling init_state() from the script, fall back to defaults
        let state = ast
            .as_ref()
            .and_then(|ast| {
                let mut scope = Scope::new();
                engine
                    .call_fn::<Dynamic>(&mut scope, ast, "init_state", ())
                    .ok()
            })
            .unwrap_or_else(default_state);

        Self {
            engine,
            ast,
            shared,
            state,
            script_path,
            last_modified,
            frame_count: 0,
        }
    }

    /// Run one frame: update shared state, execute script, replay draw ops.
    pub fn run_frame(
        &mut self,
        input: &jumpblocks_ui::UiInputState,
        data: &GameUiData,
        canvas: &mut Canvas,
        event_tx: &crossbeam_channel::Sender<GameUiEvent>,
    ) {
        // 1. Prepare shared state for this frame
        {
            let mut s = self.shared.lock().unwrap();
            s.draw_ops.clear();
            s.events.clear();
            s.keys_just_pressed.clear();

            s.window_size = [input.window_size.x, input.window_size.y];
            for code in &input.keys_just_pressed {
                s.keys_just_pressed.insert(keycode_to_string(*code));
            }

            s.game_data = game_data_to_dynamic(data);
        }

        // 2. Hot reload check (~1 second at 120fps)
        self.frame_count += 1;
        if self.frame_count % 120 == 0 {
            self.check_hot_reload();
        }

        // 3. Execute script's draw(state) function
        if let Some(ast) = &self.ast {
            let mut scope = Scope::new();
            match self
                .engine
                .call_fn::<Dynamic>(&mut scope, ast, "draw", (self.state.clone(),))
            {
                Ok(new_state) => {
                    self.state = new_state;
                }
                Err(e) => {
                    if self.frame_count % 120 == 1 {
                        eprintln!("[rhai] Runtime error: {e}");
                    }
                }
            }
        }

        // 4. Replay draw ops onto the real Canvas
        let mut s = self.shared.lock().unwrap();
        for op in &s.draw_ops {
            match op {
                DrawOp::Rect { x, y, w, h, color } => {
                    canvas.rect(*x, *y, *w, *h, *color);
                }
                DrawOp::RectFfd {
                    x,
                    y,
                    w,
                    h,
                    color,
                    ffd_id,
                } => {
                    if let Some(ffd) = s.ffd_sims.get(ffd_id) {
                        canvas.rect_ffd(*x, *y, *w, *h, *color, ffd);
                    }
                }
                DrawOp::Text {
                    x,
                    y,
                    text,
                    font_size,
                    color,
                } => {
                    canvas.text(*x, *y, text, *font_size, *color);
                }
                DrawOp::TextFfd {
                    x,
                    y,
                    text,
                    font_size,
                    color,
                    ffd_id,
                } => {
                    if let Some(ffd) = s.ffd_sims.get(ffd_id) {
                        canvas.text_ffd(*x, *y, text, *font_size, *color, ffd);
                    }
                }
                DrawOp::PushClip { x, y, w, h } => {
                    canvas.push_clip(*x, *y, *w, *h);
                }
                DrawOp::PopClip => {
                    canvas.pop_clip();
                }
            }
        }

        // 5. Send events to game thread
        let events: Vec<_> = s.events.drain(..).collect();
        drop(s);
        for event in events {
            let _ = event_tx.send(event);
        }
    }

    fn check_hot_reload(&mut self) {
        let new_mtime = file_mtime(&self.script_path);
        if new_mtime != self.last_modified {
            self.last_modified = new_mtime;
            match load_script(&self.engine, &self.script_path) {
                Some(ast) => {
                    eprintln!(
                        "[rhai] Hot-reloaded: {} (state preserved)",
                        self.script_path.display()
                    );
                    self.ast = Some(ast);
                }
                None => {
                    // Keep old AST — compile error already logged
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Engine construction — register all native functions
// ---------------------------------------------------------------------------

fn build_engine(shared: Arc<Mutex<Shared>>) -> Engine {
    let mut engine = Engine::new();

    // --- Math helpers ---
    engine.register_fn("min", |a: INT, b: INT| -> INT { a.min(b) });
    engine.register_fn("max", |a: INT, b: INT| -> INT { a.max(b) });
    engine.register_fn("clamp", |v: FLOAT, lo: FLOAT, hi: FLOAT| -> FLOAT {
        v.clamp(lo, hi)
    });

    // --- Window ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn("win_w", move || -> FLOAT {
            s.lock().unwrap().window_size[0] as f64
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("win_h", move || -> FLOAT {
            s.lock().unwrap().window_size[1] as f64
        });
    }

    // --- Game data ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn("game_data", move || -> Dynamic {
            s.lock().unwrap().game_data.clone()
        });
    }

    // --- Input ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn("key_just_pressed", move |key: &str| -> bool {
            s.lock().unwrap().keys_just_pressed.contains(key)
        });
    }

    // --- Draw: rect ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "rect",
            move |x: FLOAT, y: FLOAT, w: FLOAT, h: FLOAT, color: Array| {
                let c = array_to_color(&color);
                s.lock().unwrap().draw_ops.push(DrawOp::Rect {
                    x: x as f32,
                    y: y as f32,
                    w: w as f32,
                    h: h as f32,
                    color: c,
                });
            },
        );
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "rect_ffd",
            move |x: FLOAT, y: FLOAT, w: FLOAT, h: FLOAT, color: Array, ffd_id: INT| {
                let c = array_to_color(&color);
                s.lock().unwrap().draw_ops.push(DrawOp::RectFfd {
                    x: x as f32,
                    y: y as f32,
                    w: w as f32,
                    h: h as f32,
                    color: c,
                    ffd_id,
                });
            },
        );
    }

    // --- Draw: text ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "text",
            move |x: FLOAT, y: FLOAT, text: &str, font_size: FLOAT, color: Array| {
                let c = array_to_color(&color);
                s.lock().unwrap().draw_ops.push(DrawOp::Text {
                    x: x as f32,
                    y: y as f32,
                    text: text.to_string(),
                    font_size: font_size as f32,
                    color: c,
                });
            },
        );
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "text_ffd",
            move |x: FLOAT, y: FLOAT, text: &str, font_size: FLOAT, color: Array, ffd_id: INT| {
                let c = array_to_color(&color);
                s.lock().unwrap().draw_ops.push(DrawOp::TextFfd {
                    x: x as f32,
                    y: y as f32,
                    text: text.to_string(),
                    font_size: font_size as f32,
                    color: c,
                    ffd_id,
                });
            },
        );
    }

    // --- Draw: clip ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "push_clip",
            move |x: FLOAT, y: FLOAT, w: FLOAT, h: FLOAT| {
                s.lock().unwrap().draw_ops.push(DrawOp::PushClip {
                    x: x as f32,
                    y: y as f32,
                    w: w as f32,
                    h: h as f32,
                });
            },
        );
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("pop_clip", move || {
            s.lock().unwrap().draw_ops.push(DrawOp::PopClip);
        });
    }

    // --- FFD operations ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "ffd_new",
            move |x: FLOAT, y: FLOAT, w: FLOAT, h: FLOAT| -> INT {
                let mut shared = s.lock().unwrap();
                let id = shared.next_ffd_id;
                shared.next_ffd_id += 1;
                let sim = FfdSim::new(x as f32, y as f32, w as f32, h as f32);
                shared.ffd_sims.insert(id, sim);
                id
            },
        );
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("ffd_destroy", move |id: INT| {
            s.lock().unwrap().ffd_sims.remove(&id);
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("ffd_step", move |id: INT, dt: FLOAT| {
            if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                ffd.step(dt as f32);
            }
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("ffd_pop", move |id: INT, strength: FLOAT| {
            if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                ffd.pop(strength as f32);
            }
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("ffd_jiggle", move |id: INT, strength: FLOAT, seed: INT| {
            if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                ffd.jiggle(strength as f32, seed as u32);
            }
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "ffd_impulse_at",
            move |id: INT, px: FLOAT, py: FLOAT, vx: FLOAT, vy: FLOAT, radius: FLOAT| {
                if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                    ffd.impulse_at(px as f32, py as f32, vx as f32, vy as f32, radius as f32);
                }
            },
        );
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "ffd_resize",
            move |id: INT, x: FLOAT, y: FLOAT, w: FLOAT, h: FLOAT| {
                if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                    ffd.resize(x as f32, y as f32, w as f32, h as f32);
                }
            },
        );
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("ffd_apply_impulse", move |id: INT, vx: FLOAT, vy: FLOAT| {
            if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                ffd.apply_impulse(vx as f32, vy as f32);
            }
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "ffd_apply_force",
            move |id: INT, fx: FLOAT, fy: FLOAT, dt: FLOAT| {
                if let Some(ffd) = s.lock().unwrap().ffd_sims.get_mut(&id) {
                    ffd.apply_force(fx as f32, fy as f32, dt as f32);
                }
            },
        );
    }

    // --- Events ---
    {
        let s = Arc::clone(&shared);
        engine.register_fn("send_inventory_closed", move || {
            s.lock()
                .unwrap()
                .events
                .push(GameUiEvent::InventoryClosed);
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("send_item_selected", move |slot: INT| {
            s.lock()
                .unwrap()
                .events
                .push(GameUiEvent::ItemSelected(slot as usize));
        });
    }

    engine
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_state() -> Dynamic {
    let mut map = Map::new();
    map.insert("selected".into(), Dynamic::from(0_i64));
    map.insert("cursor_pulse".into(), Dynamic::from(0.0_f64));
    map.insert("phase".into(), Dynamic::from("hidden".to_string()));
    map.insert("scale".into(), Dynamic::from(0.6_f64));
    Dynamic::from(map)
}

fn load_script(engine: &Engine, path: &Path) -> Option<AST> {
    match std::fs::read_to_string(path) {
        Ok(source) => match engine.compile(&source) {
            Ok(ast) => {
                eprintln!("[rhai] Loaded: {}", path.display());
                Some(ast)
            }
            Err(e) => {
                eprintln!("[rhai] Compile error in {}: {e}", path.display());
                None
            }
        },
        Err(e) => {
            eprintln!("[rhai] Failed to read {}: {e}", path.display());
            None
        }
    }
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
}

fn array_to_color(arr: &Array) -> [f32; 4] {
    [
        arr.first()
            .and_then(|v| v.as_float().ok())
            .unwrap_or(1.0) as f32,
        arr.get(1)
            .and_then(|v| v.as_float().ok())
            .unwrap_or(1.0) as f32,
        arr.get(2)
            .and_then(|v| v.as_float().ok())
            .unwrap_or(1.0) as f32,
        arr.get(3)
            .and_then(|v| v.as_float().ok())
            .unwrap_or(1.0) as f32,
    ]
}

fn keycode_to_string(code: KeyCode) -> String {
    match code {
        KeyCode::ArrowUp => "ArrowUp",
        KeyCode::ArrowDown => "ArrowDown",
        KeyCode::ArrowLeft => "ArrowLeft",
        KeyCode::ArrowRight => "ArrowRight",
        KeyCode::Enter => "Enter",
        KeyCode::Escape => "Escape",
        KeyCode::Tab => "Tab",
        KeyCode::Space => "Space",
        KeyCode::Backspace => "Backspace",
        _ => return format!("{code:?}"),
    }
    .to_string()
}

fn game_data_to_dynamic(data: &GameUiData) -> Dynamic {
    let mut map = Map::new();
    map.insert("inventory_open".into(), Dynamic::from(data.inventory_open));
    map.insert("health".into(), Dynamic::from(data.health as f64));
    map.insert("max_health".into(), Dynamic::from(data.max_health as f64));

    let items: Array = data
        .items
        .iter()
        .map(|item| {
            let mut m = Map::new();
            m.insert("name".into(), Dynamic::from(item.name.clone()));
            m.insert(
                "description".into(),
                Dynamic::from(item.description.clone()),
            );
            let color: Array = item
                .color
                .iter()
                .map(|&c| Dynamic::from(c as f64))
                .collect();
            m.insert("color".into(), Dynamic::from(color));
            Dynamic::from(m)
        })
        .collect();
    map.insert("items".into(), Dynamic::from(items));

    Dynamic::from(map)
}
