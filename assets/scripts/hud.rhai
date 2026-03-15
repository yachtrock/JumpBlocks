// hud.rhai — Main HUD entry point
//
// Hot-reloadable! Edit any .rhai file in assets/scripts/ and changes
// take effect within ~1 second. Script state is preserved across reloads.
//
// The engine merges all .rhai files in this directory, so functions
// defined in inventory.rhai (and future modules) are directly callable.

/// Called once on first load to initialize persistent state.
/// Collects state from each module into a single map.
fn init_state() {
    #{
        inventory: init_inventory_state()
    }
}

/// Called every frame (~120fps) by the UI thread.
fn draw(state) {
    let ww = win_w();
    let wh = win_h();
    let data = game_data();

    // --- Always draw HUD ---
    draw_hud(data, ww, wh);

    // --- Inventory panel ---
    state.inventory = draw_inventory(state.inventory, data, ww, wh);

    state
}

/// HUD overlay — always visible regardless of inventory state.
fn draw_hud(data, ww, wh) {
    let pct = if data.max_health > 0.0 {
        data.health / data.max_health
    } else {
        0.0
    };

    // Health bar background
    rect(20.0, 20.0, 200.0, 24.0, [0.1, 0.1, 0.1, 0.8]);
    // Health bar fill
    rect(22.0, 22.0, 196.0 * pct, 20.0, [0.2, 0.9, 0.3, 1.0]);
    // Health text
    let hp_text = "HP: " + data.health.to_int() + " / " + data.max_health.to_int();
    text(28.0, 23.0, hp_text, 16.0, [1.0, 1.0, 1.0, 1.0]);

    // Bottom-right label
    text(ww - 140.0, wh - 30.0, "JumpBlocks", 14.0, [1.0, 1.0, 1.0, 0.5]);

    // Inventory hint
    if !data.inventory_open {
        text(20.0, 54.0, "[TAB] Inventory", 12.0, [0.7, 0.7, 0.7, 0.6]);
    }
}
