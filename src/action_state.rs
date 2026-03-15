//! Rhai-scripted player action states.
//!
//! Action states represent what the player is *doing* (building, combat, etc.),
//! as opposed to `PlayerState` which tracks movement (Idle/Walk/Run/Jump/Fall).
//! Each state is defined by a `.rhai` script in `assets/action_states/` with
//! `on_enter`, `on_update`, and `on_exit` callbacks.
//!
//! States can:
//! - Override inputs (declare which buttons they consume)
//! - Carry replicated variables (synced via `ActionStateVars` component)
//! - Hot-reload while the game is running

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use bevy::input::keyboard::KeyCode;
use bevy::prelude::*;
use rhai::{Dynamic, Engine, Map, Scope, AST, INT};
use serde::{Deserialize, Serialize};

use crate::player::Player;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// The identity of the current action state. `None` means default (no override).
/// This is the networked part — just a string name.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionState(pub Option<String>);

/// A serializable value for replicated state variables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StateValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// Per-state replicated variables. Only present on the entity while in a state.
/// Inserted on state enter, updated each frame, removed on state exit.
#[derive(Component, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionStateVars(pub HashMap<String, StateValue>);

/// Set of input names the current action state wants to consume this frame.
/// Checked by `player_input` to skip default bindings.
#[derive(Component, Clone, Debug, Default)]
pub struct ConsumedInputs(pub HashSet<String>);

// ---------------------------------------------------------------------------
// Enter state requests (resource-based queue)
// ---------------------------------------------------------------------------

/// A request to enter an action state.
#[derive(Clone, Debug)]
pub struct EnterActionStateRequest {
    pub state_name: String,
    pub context: HashMap<String, StateValue>,
}

/// Resource: queue of pending enter-state requests.
#[derive(Resource, Default)]
pub struct EnterActionStateQueue(pub Vec<EnterActionStateRequest>);

// ---------------------------------------------------------------------------
// Button hints
// ---------------------------------------------------------------------------

/// A single button hint: label + the key name for keyboard and gamepad.
/// For compound hints (e.g. "← →" for rotate), use `keyboard2`/`gamepad2`.
#[derive(Clone, Debug)]
pub struct ButtonHint {
    pub label: String,
    pub keyboard: String,
    pub gamepad: String,
    /// Optional second key shown alongside the first (e.g. left+right dpad).
    pub keyboard2: Option<String>,
    pub gamepad2: Option<String>,
}

/// Current button hints, set by the active action state script.
#[derive(Resource, Clone, Debug, Default)]
pub struct ButtonHints(pub Vec<ButtonHint>);

/// Events emitted by action state scripts this frame.
/// Filled by `action_state_update`, consumed by gameplay systems (e.g. building).
#[derive(Resource, Clone, Debug, Default)]
pub struct ActionStateEmits(pub Vec<String>);

/// Last input source: keyboard or gamepad. Updated each frame based on
/// which source produced input ("last-input-wins").
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub enum InputMode {
    #[default]
    Keyboard,
    Gamepad,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Per-frame snapshot of all gamepad/keyboard inputs as string names.
#[derive(Resource, Default)]
pub struct ActionStateInput {
    buttons_just_pressed: HashSet<String>,
    buttons_pressed: HashSet<String>,
    /// Whether any gamepad input was detected this frame.
    had_gamepad_input: bool,
    /// Whether any keyboard input was detected this frame.
    had_keyboard_input: bool,
}

/// Shared mutable state accessed by Rhai-registered closures via Arc<Mutex>.
struct Shared {
    consumed_inputs: HashSet<String>,
    replicated_vars: HashMap<String, StateValue>,
    pending_exit: bool,
    pending_emits: Vec<String>,
    input_just_pressed: HashSet<String>,
    input_pressed: HashSet<String>,
    button_hints: Vec<(String, String, String, Option<String>, Option<String>)>,
}

impl Shared {
    fn new() -> Self {
        Self {
            consumed_inputs: HashSet::new(),
            replicated_vars: HashMap::new(),
            pending_exit: false,
            pending_emits: Vec::new(),
            input_just_pressed: HashSet::new(),
            input_pressed: HashSet::new(),
            button_hints: Vec::new(),
        }
    }

    fn clear_frame(&mut self) {
        self.consumed_inputs.clear();
        self.replicated_vars.clear();
        self.pending_exit = false;
        self.pending_emits.clear();
        // NOTE: button_hints are NOT cleared per-frame — they persist until
        // the script calls clear_button_hints() or the state exits.
    }
}

/// Rhai engine for action state scripts. Runs on the game thread.
#[derive(Resource)]
pub struct ActionStateEngine {
    engine: Engine,
    /// Map of state name -> compiled AST.
    states: HashMap<String, AST>,
    /// Persistent Rhai state data (survives frame-to-frame, reset on state exit).
    state_data: Dynamic,
    shared: Arc<Mutex<Shared>>,
    /// Directory containing action state `.rhai` scripts.
    script_dir: PathBuf,
    /// Tracked files for hot-reload.
    file_mtimes: Vec<(PathBuf, Option<SystemTime>)>,
    frame_count: u64,
}

impl ActionStateEngine {
    pub fn new(script_dir: impl Into<PathBuf>) -> Self {
        let shared = Arc::new(Mutex::new(Shared::new()));
        let script_dir = script_dir.into();
        let engine = build_engine(Arc::clone(&shared));

        let (states, file_mtimes) = load_all_states(&engine, &script_dir);

        Self {
            engine,
            states,
            state_data: Dynamic::UNIT,
            shared,
            script_dir,
            file_mtimes,
            frame_count: 0,
        }
    }

    /// Enter a state: compile if needed, call on_enter, return initial vars.
    fn enter_state(
        &mut self,
        state_name: &str,
        context: &HashMap<String, StateValue>,
    ) -> Option<HashMap<String, StateValue>> {
        let ast = self.states.get(state_name)?;

        // Build context map for Rhai
        let mut ctx = Map::new();
        for (k, v) in context {
            ctx.insert(k.as_str().into(), state_value_to_dynamic(v));
        }

        // Clear shared state for the on_enter call
        {
            let mut s = self.shared.lock().unwrap();
            s.clear_frame();
        }

        // Call on_enter(ctx)
        let mut scope = Scope::new();
        match self
            .engine
            .call_fn::<Dynamic>(&mut scope, ast, "on_enter", (Dynamic::from(ctx),))
        {
            Ok(data) => {
                self.state_data = data;
            }
            Err(e) => {
                eprintln!("[action_state] on_enter error for '{state_name}': {e}");
                self.state_data = Dynamic::UNIT;
            }
        }

        // Collect replicated vars set during on_enter
        let vars = {
            let s = self.shared.lock().unwrap();
            if s.replicated_vars.is_empty() {
                None
            } else {
                Some(s.replicated_vars.clone())
            }
        };

        Some(vars.unwrap_or_default())
    }

    /// Run one frame of the active state. Returns (consumed_inputs, updated_vars, should_exit, emits).
    fn update_state(
        &mut self,
        state_name: &str,
        input: &ActionStateInput,
    ) -> (
        HashSet<String>,
        HashMap<String, StateValue>,
        bool,
        Vec<String>,
    ) {
        let empty = || {
            (
                HashSet::new(),
                HashMap::new(),
                false,
                Vec::new(),
            )
        };

        let Some(ast) = self.states.get(state_name) else {
            return empty();
        };

        // Prepare shared state
        {
            let mut s = self.shared.lock().unwrap();
            s.clear_frame();
            s.input_just_pressed.clone_from(&input.buttons_just_pressed);
            s.input_pressed.clone_from(&input.buttons_pressed);
        }

        // Call on_update(state, input)
        let mut scope = Scope::new();
        match self.engine.call_fn::<Dynamic>(
            &mut scope,
            ast,
            "on_update",
            (self.state_data.clone(), Dynamic::UNIT),
        ) {
            Ok(new_state) => {
                self.state_data = new_state;
            }
            Err(e) => {
                if self.frame_count % 120 == 0 {
                    eprintln!("[action_state] on_update error for '{state_name}': {e}");
                }
            }
        }

        // Collect results from shared
        let s = self.shared.lock().unwrap();
        (
            s.consumed_inputs.clone(),
            s.replicated_vars.clone(),
            s.pending_exit,
            s.pending_emits.clone(),
        )
    }

    /// Call on_exit for the current state.
    fn exit_state(&mut self, state_name: &str) {
        if let Some(ast) = self.states.get(state_name) {
            let mut scope = Scope::new();
            if let Err(e) = self.engine.call_fn::<Dynamic>(
                &mut scope,
                ast,
                "on_exit",
                (self.state_data.clone(),),
            ) {
                eprintln!("[action_state] on_exit error for '{state_name}': {e}");
            }
        }
        self.state_data = Dynamic::UNIT;
    }

    /// Check for hot-reload of state scripts.
    fn check_hot_reload(&mut self) {
        let current_files = collect_rhai_files(&self.script_dir);
        let changed = current_files.len() != self.file_mtimes.len()
            || current_files
                .iter()
                .zip(self.file_mtimes.iter())
                .any(|((pa, ma), (pb, mb))| pa != pb || ma != mb);

        if changed {
            let (new_states, new_mtimes) = load_all_states(&self.engine, &self.script_dir);
            self.file_mtimes = new_mtimes;
            // Merge: update changed scripts, keep state_data for active state
            for (name, ast) in new_states {
                self.states.insert(name, ast);
            }
            eprintln!(
                "[action_state] Hot-reloaded scripts from {}",
                self.script_dir.display()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------

fn build_engine(shared: Arc<Mutex<Shared>>) -> Engine {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(128, 64);

    // consume_input(name)
    {
        let s = Arc::clone(&shared);
        engine.register_fn("consume_input", move |name: &str| {
            s.lock().unwrap().consumed_inputs.insert(name.to_string());
        });
    }

    // exit_state()
    {
        let s = Arc::clone(&shared);
        engine.register_fn("exit_state", move || {
            s.lock().unwrap().pending_exit = true;
        });
    }

    // emit(name)
    {
        let s = Arc::clone(&shared);
        engine.register_fn("emit", move |name: &str| {
            s.lock().unwrap().pending_emits.push(name.to_string());
        });
    }

    // replicate(name, value) — multiple overloads for different Rhai types
    {
        let s = Arc::clone(&shared);
        engine.register_fn("replicate", move |name: &str, value: INT| {
            s.lock()
                .unwrap()
                .replicated_vars
                .insert(name.to_string(), StateValue::Int(value));
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("replicate", move |name: &str, value: f64| {
            s.lock()
                .unwrap()
                .replicated_vars
                .insert(name.to_string(), StateValue::Float(value));
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("replicate", move |name: &str, value: &str| {
            s.lock()
                .unwrap()
                .replicated_vars
                .insert(name.to_string(), StateValue::Str(value.to_string()));
        });
    }
    {
        let s = Arc::clone(&shared);
        engine.register_fn("replicate", move |name: &str, value: bool| {
            s.lock()
                .unwrap()
                .replicated_vars
                .insert(name.to_string(), StateValue::Bool(value));
        });
    }

    // input_just_pressed(name)
    {
        let s = Arc::clone(&shared);
        engine.register_fn("input_just_pressed", move |name: &str| -> bool {
            s.lock().unwrap().input_just_pressed.contains(name)
        });
    }

    // input_pressed(name)
    {
        let s = Arc::clone(&shared);
        engine.register_fn("input_pressed", move |name: &str| -> bool {
            s.lock().unwrap().input_pressed.contains(name)
        });
    }

    // log(msg)
    engine.register_fn("log", |msg: &str| {
        info!("[action_state] {msg}");
    });

    // set_button_hint(label, keyboard, gamepad)
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "set_button_hint",
            move |label: &str, keyboard: &str, gamepad: &str| {
                s.lock().unwrap().button_hints.push((
                    label.to_string(),
                    keyboard.to_string(),
                    gamepad.to_string(),
                    None,
                    None,
                ));
            },
        );
    }

    // set_button_hint_pair(label, keyboard1, keyboard2, gamepad1, gamepad2)
    // For compound hints like "← →" for rotate
    {
        let s = Arc::clone(&shared);
        engine.register_fn(
            "set_button_hint_pair",
            move |label: &str, kb1: &str, kb2: &str, gp1: &str, gp2: &str| {
                s.lock().unwrap().button_hints.push((
                    label.to_string(),
                    kb1.to_string(),
                    gp1.to_string(),
                    Some(kb2.to_string()),
                    Some(gp2.to_string()),
                ));
            },
        );
    }

    // clear_button_hints()
    {
        let s = Arc::clone(&shared);
        engine.register_fn("clear_button_hints", move || {
            s.lock().unwrap().button_hints.clear();
        });
    }

    engine
}

// ---------------------------------------------------------------------------
// Script loading
// ---------------------------------------------------------------------------

fn collect_rhai_files(dir: &Path) -> Vec<(PathBuf, Option<SystemTime>)> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "rhai"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let mtime = std::fs::metadata(&p)
                .ok()
                .and_then(|m| m.modified().ok());
            (p, mtime)
        })
        .collect()
}

/// Load all `.rhai` files from the action states directory.
/// Returns a map of state name (filename stem) -> compiled AST.
fn load_all_states(engine: &Engine, dir: &Path) -> (HashMap<String, AST>, Vec<(PathBuf, Option<SystemTime>)>) {
    let file_mtimes = collect_rhai_files(dir);
    let mut states = HashMap::new();

    for (path, _) in &file_mtimes {
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[action_state] Failed to read {}: {e}", path.display());
                continue;
            }
        };
        match engine.compile(&source) {
            Ok(ast) => {
                eprintln!("[action_state] Compiled: {name}");
                states.insert(name.to_string(), ast);
            }
            Err(e) => {
                eprintln!("[action_state] Compile error in {}: {e}", path.display());
            }
        }
    }

    (states, file_mtimes)
}

fn state_value_to_dynamic(v: &StateValue) -> Dynamic {
    match v {
        StateValue::Int(i) => Dynamic::from(*i),
        StateValue::Float(f) => Dynamic::from(*f),
        StateValue::Str(s) => Dynamic::from(s.clone()),
        StateValue::Bool(b) => Dynamic::from(*b),
    }
}

// ---------------------------------------------------------------------------
// Input name mapping
// ---------------------------------------------------------------------------

fn gamepad_button_name(button: GamepadButton) -> &'static str {
    match button {
        GamepadButton::South => "GamepadSouth",
        GamepadButton::East => "GamepadEast",
        GamepadButton::West => "GamepadWest",
        GamepadButton::North => "GamepadNorth",
        GamepadButton::LeftTrigger => "GamepadLeftTrigger",
        GamepadButton::LeftTrigger2 => "GamepadLeftTrigger2",
        GamepadButton::RightTrigger => "GamepadRightTrigger",
        GamepadButton::RightTrigger2 => "GamepadRightTrigger2",
        GamepadButton::Select => "GamepadSelect",
        GamepadButton::Start => "GamepadStart",
        GamepadButton::LeftThumb => "GamepadLeftThumb",
        GamepadButton::RightThumb => "GamepadRightThumb",
        GamepadButton::DPadUp => "GamepadDPadUp",
        GamepadButton::DPadDown => "GamepadDPadDown",
        GamepadButton::DPadLeft => "GamepadDPadLeft",
        GamepadButton::DPadRight => "GamepadDPadRight",
        _ => "GamepadOther",
    }
}

fn keycode_name(code: KeyCode) -> String {
    match code {
        KeyCode::Escape => "Escape".to_string(),
        KeyCode::Enter => "Enter".to_string(),
        KeyCode::Space => "Space".to_string(),
        KeyCode::Tab => "Tab".to_string(),
        KeyCode::Backspace => "Backspace".to_string(),
        KeyCode::ArrowUp => "ArrowUp".to_string(),
        KeyCode::ArrowDown => "ArrowDown".to_string(),
        KeyCode::ArrowLeft => "ArrowLeft".to_string(),
        KeyCode::ArrowRight => "ArrowRight".to_string(),
        KeyCode::ShiftLeft | KeyCode::ShiftRight => "Shift".to_string(),
        KeyCode::ControlLeft | KeyCode::ControlRight => "Control".to_string(),
        other => format!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Captures gamepad + keyboard into ActionStateInput resource and updates InputMode.
fn action_state_input_snapshot(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut input: ResMut<ActionStateInput>,
    mut input_mode: ResMut<InputMode>,
) {
    input.buttons_just_pressed.clear();
    input.buttons_pressed.clear();
    input.had_keyboard_input = false;
    input.had_gamepad_input = false;

    // Keyboard
    for code in keyboard.get_just_pressed() {
        input.buttons_just_pressed.insert(keycode_name(*code));
        input.had_keyboard_input = true;
    }
    for code in keyboard.get_pressed() {
        input.buttons_pressed.insert(keycode_name(*code));
    }

    // Gamepad
    let all_buttons = [
        GamepadButton::South,
        GamepadButton::East,
        GamepadButton::West,
        GamepadButton::North,
        GamepadButton::LeftTrigger,
        GamepadButton::LeftTrigger2,
        GamepadButton::RightTrigger,
        GamepadButton::RightTrigger2,
        GamepadButton::Select,
        GamepadButton::Start,
        GamepadButton::DPadUp,
        GamepadButton::DPadDown,
        GamepadButton::DPadLeft,
        GamepadButton::DPadRight,
    ];

    for gamepad in gamepads.iter() {
        for &btn in &all_buttons {
            let name = gamepad_button_name(btn);
            if gamepad.just_pressed(btn) {
                input.buttons_just_pressed.insert(name.to_string());
                input.had_gamepad_input = true;
            }
            if gamepad.pressed(btn) {
                input.buttons_pressed.insert(name.to_string());
            }
        }
    }

    // Last-input-wins for InputMode
    if input.had_gamepad_input {
        *input_mode = InputMode::Gamepad;
    } else if input.had_keyboard_input {
        *input_mode = InputMode::Keyboard;
    }
}

/// Runs the active state's on_update(), handles enter/exit transitions.
pub fn action_state_update(
    mut commands: Commands,
    mut engine: ResMut<ActionStateEngine>,
    input: Res<ActionStateInput>,
    mut enter_queue: ResMut<EnterActionStateQueue>,
    mut button_hints: ResMut<ButtonHints>,
    mut action_emits: ResMut<ActionStateEmits>,
    mut player_query: Query<
        (Entity, &mut ActionState, &mut ConsumedInputs, Option<&ActionStateVars>),
        With<Player>,
    >,
) {
    // Hot-reload check
    engine.frame_count += 1;
    if engine.frame_count % 120 == 0 {
        engine.check_hot_reload();
    }

    let Ok((entity, mut action_state, mut consumed, has_vars)) = player_query.single_mut() else {
        enter_queue.0.clear();
        return;
    };

    // Handle enter requests (takes priority — exit current state first if needed)
    let requests: Vec<_> = enter_queue.0.drain(..).collect();
    for request in requests {
        // Exit current state if any
        if let Some(ref current_name) = action_state.0 {
            engine.exit_state(current_name);
            commands.entity(entity).remove::<ActionStateVars>();
        }

        // Enter new state
        if let Some(initial_vars) = engine.enter_state(&request.state_name, &request.context) {
            action_state.0 = Some(request.state_name.clone());
            commands
                .entity(entity)
                .insert(ActionStateVars(initial_vars));
            info!("[action_state] Entered state: {}", request.state_name);
        } else {
            eprintln!(
                "[action_state] Failed to enter state '{}': script not found",
                request.state_name
            );
            action_state.0 = None;
        }
    }

    // Run on_update for active state
    if let Some(ref state_name) = action_state.0.clone() {
        let (new_consumed, updated_vars, should_exit, emits) =
            engine.update_state(state_name, &input);

        consumed.0 = new_consumed;

        // Update replicated vars if changed
        if !updated_vars.is_empty() {
            if has_vars.is_some() {
                // Update existing component
                commands
                    .entity(entity)
                    .insert(ActionStateVars(updated_vars));
            }
        }

        // Store emits for gameplay systems to consume
        action_emits.0 = emits;

        // Copy button hints from shared → resource
        {
            let s = engine.shared.lock().unwrap();
            button_hints.0 = s
                .button_hints
                .iter()
                .map(|(label, kb, gp, kb2, gp2)| ButtonHint {
                    label: label.clone(),
                    keyboard: kb.clone(),
                    gamepad: gp.clone(),
                    keyboard2: kb2.clone(),
                    gamepad2: gp2.clone(),
                })
                .collect();
        }

        // Handle exit
        if should_exit {
            engine.exit_state(state_name);
            action_state.0 = None;
            consumed.0.clear();
            commands.entity(entity).remove::<ActionStateVars>();
            // Auto-clear button hints on state exit
            button_hints.0.clear();
            {
                engine.shared.lock().unwrap().button_hints.clear();
            }
            info!("[action_state] Exited state: {state_name}");
        }
    } else {
        consumed.0.clear();
        button_hints.0.clear();
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct ActionStatePlugin;

impl Plugin for ActionStatePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ActionStateInput::default());
        app.insert_resource(ActionStateEngine::new("assets/action_states"));
        app.insert_resource(EnterActionStateQueue::default());
        app.insert_resource(ButtonHints::default());
        app.insert_resource(ActionStateEmits::default());
        app.insert_resource(InputMode::default());

        app.add_systems(
            Update,
            action_state_input_snapshot.before(action_state_update),
        );
        app.add_systems(
            Update,
            action_state_update.before(crate::player::player_input),
        );
    }
}
