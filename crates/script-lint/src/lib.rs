//! Rhai script linter for JumpBlocks.
//!
//! Builds a stub Engine with the same function signatures registered in
//! `scripting.rs` and compiles all `.rhai` scripts. Catches syntax errors,
//! bad imports, and unknown function signatures without needing the full
//! game binary or system libraries.
//!
//! Run with: `cargo test -p jumpblocks-script-lint`

use rhai::module_resolvers::FileModuleResolver;
use rhai::{Array, Dynamic, Engine, FLOAT, INT};
use std::path::Path;

/// Build a lightweight engine with stub implementations of every native
/// function registered in `scripting::build_engine`. Same signatures,
/// no-op bodies.
pub fn stub_engine(script_dir: &Path) -> Engine {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(128, 64);
    engine.set_module_resolver(FileModuleResolver::new_with_path(script_dir));

    // Math
    engine.register_fn("min", |a: INT, b: INT| -> INT { a.min(b) });
    engine.register_fn("max", |a: INT, b: INT| -> INT { a.max(b) });
    engine.register_fn("clamp", |v: FLOAT, lo: FLOAT, hi: FLOAT| -> FLOAT {
        v.clamp(lo, hi)
    });

    // Window
    engine.register_fn("win_w", || -> FLOAT { 800.0 });
    engine.register_fn("win_h", || -> FLOAT { 600.0 });

    // Game data
    engine.register_fn("game_data", || -> Dynamic { Dynamic::UNIT });

    // Input
    engine.register_fn("key_just_pressed", |_key: &str| -> bool { false });

    // Draw
    engine.register_fn("rect", |_x: FLOAT, _y: FLOAT, _w: FLOAT, _h: FLOAT, _c: Array| {});
    engine.register_fn("rect_ffd", |_x: FLOAT, _y: FLOAT, _w: FLOAT, _h: FLOAT, _c: Array, _f: INT| {});
    engine.register_fn("text", |_x: FLOAT, _y: FLOAT, _t: &str, _s: FLOAT, _c: Array| {});
    engine.register_fn("text_ffd", |_x: FLOAT, _y: FLOAT, _t: &str, _s: FLOAT, _c: Array, _f: INT| {});
    engine.register_fn("push_clip", |_x: FLOAT, _y: FLOAT, _w: FLOAT, _h: FLOAT| {});
    engine.register_fn("pop_clip", || {});

    // FFD
    engine.register_fn("ffd_new", |_x: FLOAT, _y: FLOAT, _w: FLOAT, _h: FLOAT| -> INT { 1 });
    engine.register_fn("ffd_destroy", |_id: INT| {});
    engine.register_fn("ffd_step", |_id: INT, _dt: FLOAT| {});
    engine.register_fn("ffd_pop", |_id: INT, _s: FLOAT| {});
    engine.register_fn("ffd_jiggle", |_id: INT, _s: FLOAT, _seed: INT| {});
    engine.register_fn("ffd_impulse_at", |_id: INT, _px: FLOAT, _py: FLOAT, _vx: FLOAT, _vy: FLOAT, _r: FLOAT| {});
    engine.register_fn("ffd_resize", |_id: INT, _x: FLOAT, _y: FLOAT, _w: FLOAT, _h: FLOAT| {});
    engine.register_fn("ffd_apply_impulse", |_id: INT, _vx: FLOAT, _vy: FLOAT| {});
    engine.register_fn("ffd_apply_force", |_id: INT, _fx: FLOAT, _fy: FLOAT, _dt: FLOAT| {});

    // Events
    engine.register_fn("send_inventory_closed", || {});
    engine.register_fn("send_item_selected", |_slot: INT| {});

    engine
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script_dir() -> std::path::PathBuf {
        // Walk up from crates/script-lint to the workspace root
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/scripts")
            .canonicalize()
            .expect("assets/scripts directory not found")
    }

    #[test]
    fn rhai_scripts_compile() {
        let dir = script_dir();
        let engine = stub_engine(&dir);

        let main_script = dir.join("hud.rhai");
        let source = std::fs::read_to_string(&main_script)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", main_script.display()));

        let scope = Scope::new();
        let ast = engine
            .compile_into_self_contained(&scope, &source)
            .unwrap_or_else(|e| panic!("Compile error in {}: {e}", main_script.display()));

        // Verify top-level eval succeeds (catches import resolution issues)
        let mut eval_scope = Scope::new();
        let _ = engine
            .eval_ast_with_scope::<Dynamic>(&mut eval_scope, &ast)
            .unwrap_or_else(|e| panic!("Top-level eval failed: {e}"));

        // Verify init_state() can be called
        let mut call_scope = eval_scope.clone();
        let _ = engine
            .call_fn::<Dynamic>(&mut call_scope, &ast, "init_state", ())
            .unwrap_or_else(|e| panic!("init_state() failed: {e}"));
    }
}
