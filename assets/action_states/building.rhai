// Building state — entered when the player selects an inventory item.
// The script drives block preview and placement via the building API:
//   build_position()              → grid position in front of player (or () if none)
//   show_preview(wx, wy, wz, f)  → show translucent block at world position
//   show_preview_in_hand(f)      → show block floating in front of player
//   hide_preview()               → hide the preview
//   place_block(shape, facing, texture) → place at the current build_position
//   rotate_facing_right(f) / rotate_facing_left(f) → facing helpers (0=N 1=E 2=S 3=W)
//
// Facing: 0=North 1=East 2=South 3=West
// Shapes: 0=cube 1=smooth_cube 2=wedge 3=smooth_wedge

fn on_enter(ctx) {
    log("Entered building state with item: " + ctx.item_name);
    replicate("block_type", ctx.item_name);
    replicate("slot", ctx.selected_slot);

    set_button_hint("Place Block", "F", "ZR");
    set_button_hint("Put Away", "Esc", "B");
    set_button_hint("Options", "Q", "ZL");

    #{
        item_name: ctx.item_name,
        selected_slot: ctx.selected_slot,
        options_held: false,
        facing: 0
    }
}

fn on_update(state, input) {
    // Consume inputs so default player bindings don't fire
    consume_input("GamepadEast");
    consume_input("GamepadRightTrigger");
    consume_input("KeyF");
    consume_input("KeyQ");
    consume_input("GamepadLeftTrigger");

    let options_held = input_pressed("GamepadLeftTrigger") || input_pressed("KeyQ");

    // ── Exit ──────────────────────────────────────────────────────────
    if input_just_pressed("GamepadEast") || input_just_pressed("Escape") {
        exit_state();
        return state;
    }

    // ── Build position & preview ─────────────────────────────────────
    let pos = build_position();

    if pos != () {
        show_preview(pos.world_x, pos.world_y, pos.world_z, state.facing);
    } else {
        show_preview_in_hand(state.facing);
    }

    // ── Place block (wedge = shape 2) ────────────────────────────────
    if input_just_pressed("GamepadRightTrigger") || input_just_pressed("KeyF") {
        if pos != () {
            place_block(2, state.facing, 1);
            log("Placed " + state.item_name + " facing " + state.facing);
        }
    }

    // ── Options panel (hold ZL / Q) ──────────────────────────────────
    if options_held {
        consume_input("GamepadDPadLeft");
        consume_input("GamepadDPadRight");
        consume_input("GamepadDPadUp");
        consume_input("GamepadDPadDown");
        consume_input("ArrowLeft");
        consume_input("ArrowRight");
        consume_input("ArrowUp");
        consume_input("ArrowDown");

        clear_button_hints();
        set_button_hint_pair("Rotate", "←", "→", "D←", "D→");
        set_button_hint_pair("Position", "↑", "↓", "D↑", "D↓");

        // Rotate
        if input_just_pressed("GamepadDPadRight") || input_just_pressed("ArrowRight") {
            state.facing = rotate_facing_right(state.facing);
            log("Facing: " + state.facing);
        }
        if input_just_pressed("GamepadDPadLeft") || input_just_pressed("ArrowLeft") {
            state.facing = rotate_facing_left(state.facing);
            log("Facing: " + state.facing);
        }

        // Position up/down — reserved for future use
        if input_just_pressed("GamepadDPadUp") || input_just_pressed("ArrowUp") {
            log("Position up");
        }
        if input_just_pressed("GamepadDPadDown") || input_just_pressed("ArrowDown") {
            log("Position down");
        }
    } else {
        // Restore normal hints when options released
        if state.options_held {
            clear_button_hints();
            set_button_hint("Place Block", "F", "ZR");
            set_button_hint("Put Away", "Esc", "B");
            set_button_hint("Options", "Q", "ZL");
        }
    }

    replicate("facing", state.facing);
    state.options_held = options_held;
    state
}

fn on_exit(state) {
    hide_preview();
    clear_button_hints();
    log("Exited building state");
}
