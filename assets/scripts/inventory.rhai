// inventory.rhai — Inventory UI with FFD jiggle physics
//
// Hot-reloadable! Edit this file while the game is running and changes
// take effect within ~1 second. Script state (selected slot, animation
// phase, FFD simulations) is preserved across reloads.

// Layout constants
const COLS = 4;
const OPEN_SPEED = 7.0;
const CLOSE_SPEED = 10.5;
const PANEL_WIDTH = 480.0;

/// Called once on first load to initialize persistent state.
fn init_state() {
    #{
        selected: 0,
        cursor_pulse: 0.0,
        phase: "hidden",
        scale: 0.6
    }
}

/// Called every frame (~120fps) by the UI thread.
fn draw(state) {
    let ww = win_w();
    let wh = win_h();
    let dt = 1.0 / 120.0;
    let data = game_data();

    // --- Always draw HUD ---
    draw_hud(data, ww, wh);

    // --- Phase transitions ---
    if data.inventory_open && state.phase == "hidden" {
        state.phase = "opening";
        state.scale = 0.6;
    }
    if !data.inventory_open && state.phase == "open" {
        state.phase = "closing";
        if state.ffd_id != () {
            ffd_pop(state.ffd_id, -12.0);
        }
    }

    // --- Scale animation ---
    if state.phase == "opening" {
        state.scale += OPEN_SPEED * dt;
        if state.scale >= 1.0 {
            state.scale = 1.0;
            state.phase = "open";
        }
    }
    if state.phase == "closing" {
        state.scale -= CLOSE_SPEED * dt;
        if state.scale <= 0.6 {
            state.phase = "hidden";
            if state.ffd_id != () {
                ffd_destroy(state.ffd_id);
                state.ffd_id = ();
            }
            return state;
        }
    }
    if state.phase == "hidden" {
        return state;
    }

    state.cursor_pulse += 0.08;
    let items = data.items;
    let num_items = items.len();
    let rows = (num_items + COLS - 1) / COLS;

    // --- Input (only when fully open) ---
    if state.phase == "open" {
        if key_just_pressed("ArrowRight") {
            state.selected = min(state.selected + 1, num_items - 1);
        }
        if key_just_pressed("ArrowLeft") && state.selected > 0 {
            state.selected -= 1;
        }
        if key_just_pressed("ArrowDown") {
            let next = state.selected + COLS;
            if next < num_items {
                state.selected = next;
            }
        }
        if key_just_pressed("ArrowUp") && state.selected >= COLS {
            state.selected -= COLS;
        }
        if key_just_pressed("Enter") {
            send_item_selected(state.selected);
        }
        if key_just_pressed("Escape") {
            send_inventory_closed();
        }
    }

    // --- Layout ---
    let full_w = PANEL_WIDTH;
    let full_h = 60.0 + rows.to_float() * 80.0 + 80.0;
    let cx = ww * 0.5;
    let cy = wh * 0.5;
    let panel_x = cx - full_w * 0.5;
    let panel_y = cy - full_h * 0.5;

    // --- FFD setup ---
    if state.ffd_id == () {
        state.ffd_id = ffd_new(panel_x, panel_y, full_w, full_h);
        ffd_pop(state.ffd_id, 20.0);
        ffd_jiggle(state.ffd_id, 8.0, 42);
    }

    // --- Localized impulse on dpad navigation ---
    if state.phase == "open" {
        let s = state.scale;
        let grid_x0 = panel_x + 16.0 * s;
        let grid_y0 = panel_y + 52.0 * s;
        let cell_w = (full_w * s - 32.0 * s - (COLS.to_float() - 1.0) * 8.0 * s) / COLS.to_float();
        let cell_h = 72.0 * s;

        let sel_col = state.selected % COLS;
        let sel_row = state.selected / COLS;
        let sel_cx = grid_x0 + sel_col.to_float() * (cell_w + 8.0 * s) + cell_w * 0.5;
        let sel_cy = grid_y0 + sel_row.to_float() * (cell_h + 8.0 * s) + cell_h * 0.5;
        let imp = 10.0;
        let imp_r = full_w * 0.4;

        if key_just_pressed("ArrowRight") {
            ffd_impulse_at(state.ffd_id, sel_cx, sel_cy, imp, 0.0, imp_r);
        }
        if key_just_pressed("ArrowLeft") {
            ffd_impulse_at(state.ffd_id, sel_cx, sel_cy, -imp, 0.0, imp_r);
        }
        if key_just_pressed("ArrowDown") {
            ffd_impulse_at(state.ffd_id, sel_cx, sel_cy, 0.0, imp, imp_r);
        }
        if key_just_pressed("ArrowUp") {
            ffd_impulse_at(state.ffd_id, sel_cx, sel_cy, 0.0, -imp, imp_r);
        }
    }

    ffd_step(state.ffd_id, dt);

    // --- Drawing ---
    let s = state.scale;
    let sw = full_w * s;
    let sh = full_h * s;
    let sx = cx - sw * 0.5;
    let sy = cy - sh * 0.5;
    let ffd = state.ffd_id;

    // Dim background (not FFD-warped)
    rect(0.0, 0.0, ww, wh, [0.0, 0.0, 0.0, 0.5]);

    // Panel background
    rect_ffd(sx, sy, sw, sh, [0.12, 0.12, 0.18, 0.95], ffd);

    // Panel border
    let bc = [0.4, 0.4, 0.6, 1.0];
    rect_ffd(sx, sy, sw, 2.0, bc, ffd);
    rect_ffd(sx, sy + sh - 2.0, sw, 2.0, bc, ffd);
    rect_ffd(sx, sy, 2.0, sh, bc, ffd);
    rect_ffd(sx + sw - 2.0, sy, 2.0, sh, bc, ffd);

    // Title
    text_ffd(sx + 16.0 * s, sy + 16.0 * s,
             "INVENTORY", 22.0 * s,
             [0.9, 0.85, 0.6, 1.0], ffd);
    text_ffd(sx + sw - 160.0 * s, sy + 20.0 * s,
             "[ESC] Close", 13.0 * s,
             [0.6, 0.6, 0.6, 1.0], ffd);

    // Item grid
    let grid_x = sx + 16.0 * s;
    let grid_y = sy + 52.0 * s;
    let cell_w = (sw - 32.0 * s - (COLS.to_float() - 1.0) * 8.0 * s) / COLS.to_float();
    let cell_h = 72.0 * s;

    for i in 0..num_items {
        let item = items[i];
        let col = i % COLS;
        let row = i / COLS;
        let cell_x = grid_x + col.to_float() * (cell_w + 8.0 * s);
        let cell_y = grid_y + row.to_float() * (cell_h + 8.0 * s);

        let is_sel = i == state.selected;
        let bg = if is_sel {
            let pulse = clamp(sin(state.cursor_pulse) * 0.15 + 0.35, 0.2, 0.5);
            [0.3, 0.3, pulse + 0.3, 0.9]
        } else {
            [0.18, 0.18, 0.24, 0.8]
        };
        rect_ffd(cell_x, cell_y, cell_w, cell_h, bg, ffd);

        // Selection border highlight
        if is_sel {
            let b = clamp(sin(state.cursor_pulse) * 0.3 + 0.7, 0.4, 1.0);
            let sel_c = [0.5, 0.5, b, 1.0];
            rect_ffd(cell_x, cell_y, cell_w, 2.0, sel_c, ffd);
            rect_ffd(cell_x, cell_y + cell_h - 2.0, cell_w, 2.0, sel_c, ffd);
            rect_ffd(cell_x, cell_y, 2.0, cell_h, sel_c, ffd);
            rect_ffd(cell_x + cell_w - 2.0, cell_y, 2.0, cell_h, sel_c, ffd);
        }

        // Item icon (colored square)
        rect_ffd(cell_x + 8.0 * s, cell_y + 8.0 * s,
                 32.0 * s, 32.0 * s, item.color, ffd);
        // Item name
        text_ffd(cell_x + 8.0 * s, cell_y + 46.0 * s,
                 item.name, 12.0 * s,
                 [0.9, 0.9, 0.9, 1.0], ffd);
    }

    // --- Detail panel ---
    if state.selected < num_items {
        let item = items[state.selected];
        let detail_y = grid_y + rows.to_float() * (cell_h + 8.0 * s) + 8.0 * s;
        rect_ffd(grid_x, detail_y, sw - 32.0 * s, 1.0,
                 [0.4, 0.4, 0.5, 0.6], ffd);
        text_ffd(grid_x, detail_y + 10.0 * s,
                 item.name, 16.0 * s,
                 [1.0, 1.0, 1.0, 1.0], ffd);
        text_ffd(grid_x, detail_y + 32.0 * s,
                 item.description, 13.0 * s,
                 [0.7, 0.7, 0.7, 1.0], ffd);
    }

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
