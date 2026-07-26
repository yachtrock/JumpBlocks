// build.rhai — Building interface: hotbar, shape picker, pause (esc) menu.
//
// Renders from the `build` submap of game_data() (see build_ui.rs) and sends
// interactions back with send_build_event(kind, value). Hot-reloadable.

/// Persistent UI-side state for the building interface.
fn init_build_state() {
    #{
        pulse: 0.0
    }
}

/// Main entry, called from hud.rhai every frame. Returns the mutated state.
fn draw_build(b, data, ww, wh) {
    b.pulse += 0.06;
    let bd = data.build;


    // Hotbar: automatic whenever the player stands in a buildable area.
    if bd.in_area && !data.inventory_open && !bd.esc_menu_open {
        draw_hotbar(b, bd, ww, wh);
    }

    if bd.shape_menu_open {
        draw_shape_menu(b, bd, ww, wh);
    }

    if bd.esc_menu_open {
        draw_esc_menu(b, bd, ww, wh);
    }

    b
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

fn hit(x, y, w, h) {
    let mx = mouse_x();
    let my = mouse_y();
    mx >= x && mx <= x + w && my >= y && my <= y + h
}

// ---------------------------------------------------------------------------
// Hotbar (Minecraft-style)
// ---------------------------------------------------------------------------

fn draw_hotbar(b, bd, ww, wh) {
    const SLOT = 52.0;
    const GAP = 5.0;

    let mats = bd.materials;
    let n = mats.len();
    if n == 0 {
        return;
    }

    let total = n.to_float() * SLOT + (n.to_float() - 1.0) * GAP;
    let x0 = (ww - total) * 0.5;
    let y0 = wh - SLOT - 16.0;

    // Bar backdrop
    rect(x0 - 8.0, y0 - 8.0, total + 16.0, SLOT + 16.0, [0.04, 0.04, 0.06, 0.72]);

    let i = 0;
    while i < n {
        let m = mats[i];
        let x = x0 + i.to_float() * (SLOT + GAP);
        let sel = i == bd.selected_slot;

        let y = if sel { y0 - 5.0 } else { y0 };

        // Slot background + border
        let bg = if sel { [0.22, 0.22, 0.28, 0.95] } else { [0.10, 0.10, 0.13, 0.85] };
        rect(x, y, SLOT, SLOT, bg);
        let bc = if sel {
            let g = clamp(sin(b.pulse) * 0.2 + 0.85, 0.6, 1.0);
            [g, g, 0.4, 1.0]
        } else {
            [0.35, 0.35, 0.42, 0.9]
        };
        let bw = if sel { 3.0 } else { 1.5 };
        rect(x, y, SLOT, bw, bc);
        rect(x, y + SLOT - bw, SLOT, bw, bc);
        rect(x, y, bw, SLOT, bc);
        rect(x + SLOT - bw, y, bw, SLOT, bc);

        // Material swatch
        rect(x + 9.0, y + 7.0, SLOT - 18.0, SLOT - 22.0, m.color);

        // Count + slot number
        text(x + 5.0, y + SLOT - 15.0, "" + m.count, 11.0, [0.95, 0.95, 0.95, 0.95]);
        text(x + SLOT - 12.0, y + 3.0, "" + (i + 1), 10.0, [0.7, 0.7, 0.7, 0.8]);

        i += 1;
    }

    // Status line above the bar: material, shape, tool.
    let mat_name = if bd.selected_slot < n { mats[bd.selected_slot].name } else { "" };
    let shape_name = if bd.shape_selected < bd.shapes.len() {
        bd.shapes[bd.shape_selected].name
    } else {
        ""
    };
    let tool_label = if bd.tool == "auto" { "Auto Shape" } else { "Direct" };
    let status = if bd.tool == "auto" {
        mat_name + "  ·  Tool: " + tool_label + " [V]  ·  [R] Shapes"
    } else {
        mat_name + "  ·  " + shape_name + " [R]  ·  Tool: " + tool_label + " [V]"
    };
    let sw = status.len().to_float() * 6.4;
    rect((ww - sw) * 0.5 - 8.0, y0 - 30.0, sw + 16.0, 20.0, [0.04, 0.04, 0.06, 0.6]);
    text((ww - sw) * 0.5, y0 - 27.0, status, 13.0, [0.95, 0.93, 0.85, 0.95]);
}

// ---------------------------------------------------------------------------
// Shape picker popup
// ---------------------------------------------------------------------------

fn draw_shape_menu(b, bd, ww, wh) {
    const COLS = 3;
    const CELL_W = 148.0;
    const CELL_H = 116.0;
    const PAD = 16.0;
    const GAP = 8.0;

    let shapes = bd.shapes;
    let n = shapes.len();
    if n == 0 {
        return;
    }
    let rows = (n + COLS - 1) / COLS;

    let w = COLS.to_float() * CELL_W + (COLS.to_float() - 1.0) * GAP + PAD * 2.0;
    let h = rows.to_float() * CELL_H + (rows.to_float() - 1.0) * GAP + PAD * 2.0 + 66.0;
    let x0 = (ww - w) * 0.5;
    let y0 = (wh - h) * 0.5;

    // Dim + panel
    rect(0.0, 0.0, ww, wh, [0.0, 0.0, 0.0, 0.45]);
    rect(x0, y0, w, h, [0.10, 0.10, 0.15, 0.97]);
    let bc = [0.45, 0.45, 0.62, 1.0];
    rect(x0, y0, w, 2.0, bc);
    rect(x0, y0 + h - 2.0, w, 2.0, bc);
    rect(x0, y0, 2.0, h, bc);
    rect(x0 + w - 2.0, y0, 2.0, h, bc);

    text(x0 + PAD, y0 + 12.0, "SHAPES", 22.0, [0.9, 0.85, 0.6, 1.0]);
    text(x0 + w - 150.0, y0 + 17.0, "[R / Esc] Close", 12.0, [0.6, 0.6, 0.6, 1.0]);

    let grid_y = y0 + 44.0;
    let i = 0;
    while i < n {
        let s = shapes[i];
        let col = i % COLS;
        let row = i / COLS;
        let cx = x0 + PAD + col.to_float() * (CELL_W + GAP);
        let cy = grid_y + row.to_float() * (CELL_H + GAP);

        let sel = i == bd.shape_selected;
        let hov = hit(cx, cy, CELL_W, CELL_H);

        let bg = if sel {
            let p = clamp(sin(b.pulse) * 0.1 + 0.3, 0.2, 0.4);
            [0.26, 0.30, p + 0.25, 0.95]
        } else if hov {
            [0.20, 0.20, 0.28, 0.9]
        } else {
            [0.15, 0.15, 0.20, 0.85]
        };
        rect(cx, cy, CELL_W, CELL_H, bg);
        if sel {
            let sc = [0.85, 0.85, 0.45, 1.0];
            rect(cx, cy, CELL_W, 2.0, sc);
            rect(cx, cy + CELL_H - 2.0, CELL_W, 2.0, sc);
            rect(cx, cy, 2.0, CELL_H, sc);
            rect(cx + CELL_W - 2.0, cy, 2.0, CELL_H, sc);
        }

        // Icon
        draw_shape_icon(s.id, cx + (CELL_W - 56.0) * 0.5, cy + 8.0, 56.0);

        // Name + cost
        let name_w = s.name.len().to_float() * 6.5;
        text(cx + (CELL_W - name_w) * 0.5, cy + CELL_H - 40.0, s.name, 13.0, [0.95, 0.95, 0.95, 1.0]);
        let cost_label = "cost " + s.cost;
        let cost_w = cost_label.len().to_float() * 5.5;
        text(cx + (CELL_W - cost_w) * 0.5, cy + CELL_H - 21.0, cost_label, 11.0, [0.75, 0.72, 0.55, 0.95]);

        if mouse_clicked() && hov {
            send_build_event("shape", i);
        }

        i += 1;
    }

    let foot = "[Arrows] Navigate   [Enter] Select   [Click] Select";
    text(x0 + PAD, y0 + h - 22.0, foot, 12.0, [0.6, 0.6, 0.65, 0.9]);
}

/// Small side-profile silhouette for each block shape, drawn with polys.
/// Slopes are drawn high-side-left. `s` is the icon box size.
fn draw_shape_icon(id, x, y, s) {
    let base = y + s;            // bottom of the icon box
    let body = [0.72, 0.78, 0.88, 1.0];
    let dark = [0.45, 0.52, 0.66, 1.0];

    if id == 0 {
        // Cube: squat full-width slab
        rect(x, y + s * 0.45, s, s * 0.55, body);
        rect(x, y + s * 0.45, s, 3.0, dark);
    } else if id == 1 {
        // Wedge (1:2): gentle ramp
        poly([x, base, x + s, base, x + s, y + s * 0.68, x, y + s * 0.25], body);
    } else if id == 2 {
        // Outer corner (1:2): slab with a small rising nose
        rect(x, y + s * 0.68, s, s * 0.32, body);
        poly([x, y + s * 0.68, x + s * 0.55, y + s * 0.68, x, y + s * 0.3], dark);
    } else if id == 3 {
        // Inner corner (1:2): ramp with a second fold
        poly([x, base, x + s, base, x + s, y + s * 0.68, x, y + s * 0.25], body);
        poly([x, y + s * 0.25, x + s, y + s * 0.68, x + s * 0.45, y + s * 0.25], dark);
    } else if id == 4 {
        // Steep wedge (1:1)
        poly([x, base, x + s, base, x + s, y + s * 0.68, x, y], body);
    } else if id == 5 {
        // Steep outer corner: slab with a tall thin nose
        rect(x, y + s * 0.68, s, s * 0.32, body);
        poly([x, y + s * 0.68, x + s * 0.42, y + s * 0.68, x, y + s * 0.05], dark);
    } else if id == 6 {
        // Steep inner corner
        poly([x, base, x + s, base, x + s, y + s * 0.68, x, y], body);
        poly([x, y, x + s, y + s * 0.68, x + s * 0.45, y + s * 0.05], dark);
    } else if id == 7 {
        // Diagonal ramp (1:2): faceted pentagon
        poly([x, base, x + s, base, x + s, y + s * 0.72, x + s * 0.45, y + s * 0.22, x, y + s * 0.45], body);
        poly([x + s * 0.45, y + s * 0.22, x + s, y + s * 0.72, x + s * 0.45, y + s * 0.72], dark);
    } else if id == 8 {
        // Steep diagonal ramp
        poly([x, base, x + s, base, x + s, y + s * 0.66, x + s * 0.42, y, x, y + s * 0.3], body);
        poly([x + s * 0.42, y, x + s, y + s * 0.66, x + s * 0.42, y + s * 0.66], dark);
    } else {
        rect(x + s * 0.2, y + s * 0.2, s * 0.6, s * 0.6, body);
    }
}

// ---------------------------------------------------------------------------
// Pause (esc) menu
// ---------------------------------------------------------------------------

fn draw_esc_menu(b, bd, ww, wh) {
    const W = 400.0;
    const H = 252.0;

    let x0 = (ww - W) * 0.5;
    let y0 = (wh - H) * 0.5;

    rect(0.0, 0.0, ww, wh, [0.0, 0.0, 0.0, 0.55]);
    rect(x0, y0, W, H, [0.10, 0.10, 0.15, 0.97]);
    let bc = [0.45, 0.45, 0.62, 1.0];
    rect(x0, y0, W, 2.0, bc);
    rect(x0, y0 + H - 2.0, W, 2.0, bc);
    rect(x0, y0, 2.0, H, bc);
    rect(x0 + W - 2.0, y0, 2.0, H, bc);

    text(x0 + 24.0, y0 + 18.0, "PAUSED", 26.0, [0.9, 0.85, 0.6, 1.0]);

    if bd.challenge_active {
        text(x0 + 24.0, y0 + 56.0, "Challenge: " + bd.challenge_name, 14.0, [0.8, 0.8, 0.9, 1.0]);
    } else {
        text(x0 + 24.0, y0 + 56.0, "No challenge running", 13.0, [0.55, 0.55, 0.6, 0.9]);
    }

    let items = ["Resume", "Cancel Challenge"];
    let item_y0 = y0 + 92.0;
    const ITEM_H = 44.0;

    let i = 0;
    while i < items.len() {
        let iy = item_y0 + i.to_float() * (ITEM_H + 8.0);
        let sel = i == bd.esc_selected;
        let enabled = i == 0 || bd.challenge_active;
        let hov = hit(x0 + 20.0, iy, W - 40.0, ITEM_H);

        let bg = if sel {
            let p = clamp(sin(b.pulse) * 0.08 + 0.28, 0.2, 0.36);
            [0.24, 0.28, p + 0.22, 0.95]
        } else if hov && enabled {
            [0.18, 0.18, 0.25, 0.9]
        } else {
            [0.14, 0.14, 0.19, 0.85]
        };
        rect(x0 + 20.0, iy, W - 40.0, ITEM_H, bg);
        if sel {
            let sc = [0.85, 0.85, 0.45, 1.0];
            rect(x0 + 20.0, iy, W - 40.0, 2.0, sc);
            rect(x0 + 20.0, iy + ITEM_H - 2.0, W - 40.0, 2.0, sc);
            rect(x0 + 20.0, iy, 2.0, ITEM_H, sc);
            rect(x0 + W - 22.0, iy, 2.0, ITEM_H, sc);
            text(x0 + 30.0, iy + 12.0, ">", 16.0, [0.9, 0.9, 0.5, 1.0]);
        }

        let tc = if enabled { [0.95, 0.95, 0.95, 1.0] } else { [0.5, 0.5, 0.55, 0.8] };
        text(x0 + 52.0, iy + 13.0, items[i], 16.0, tc);

        if mouse_clicked() && hov && enabled {
            if i == 0 {
                send_build_event("esc_resume", 0);
            } else {
                send_build_event("esc_cancel", 0);
            }
        }

        i += 1;
    }

    text(x0 + 24.0, y0 + H - 26.0, "[Up/Down] Navigate   [Enter] Select   [Esc] Resume", 11.0, [0.6, 0.6, 0.65, 0.9]);
}
