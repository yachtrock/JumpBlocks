// Building state — auto-entered while the player stands in a buildable area
// (see build_ui.rs). The build UI owns material/shape/tool selection; this
// script drives the preview and placement each frame via the building API:
//   build_position()              → #{ x, y, z } chunk-local voxel coords (or () if none)
//   show_preview(facing)         → translucent preview of the selected shape
//   show_preview_in_hand(facing) → preview floating in front of the player
//   hide_preview()               → hide the preview
//   place_block(shape, facing, texture) → place at the current build_position
//   selected_shape()             → shape chosen in the build UI (cube in auto mode)
//   selected_texture()           → material chosen in the hotbar
//   auto_shape_mode()            → true when the auto-shape tool is active
//   in_build_area()              → false the moment the player leaves the area
//   rotate_facing_right(f) / rotate_facing_left(f) → facing helpers (0=N 1=E 2=S 3=W)

fn on_enter(ctx) {
    log("Entered building state");

    set_button_hint("Place", "F", "ZR");
    set_button_hint("Shapes", "R", "Y");
    set_button_hint("Tool", "V", "ZL");
    set_button_hint_pair("Rotate (hold Q)", "←", "→", "D←", "D→");

    #{
        options_held: false,
        facing: 0
    }
}

fn on_update(state, input) {
    // Leaving the buildable area (or opening a menu) puts the tools away.
    if !in_build_area() {
        exit_state();
        return state;
    }

    // Consume inputs so default player bindings don't fire
    consume_input("GamepadRightTrigger");
    consume_input("KeyF");
    consume_input("KeyQ");
    consume_input("GamepadLeftTrigger");

    let options_held = input_pressed("GamepadLeftTrigger") || input_pressed("KeyQ");

    // ── Build position & preview ─────────────────────────────────────
    let pos = build_position();

    if pos != () {
        show_preview(state.facing);
    } else {
        show_preview_in_hand(state.facing);
    }

    // ── Place the selected shape with the selected material ──────────
    if input_just_pressed("GamepadRightTrigger") || input_just_pressed("KeyF") {
        if pos != () {
            place_block(selected_shape(), state.facing, selected_texture());
        }
    }

    // ── Rotate (hold Q / ZL + arrows or dpad) ────────────────────────
    if options_held {
        consume_input("GamepadDPadLeft");
        consume_input("GamepadDPadRight");
        consume_input("ArrowLeft");
        consume_input("ArrowRight");

        if input_just_pressed("GamepadDPadRight") || input_just_pressed("ArrowRight") {
            state.facing = rotate_facing_right(state.facing);
        }
        if input_just_pressed("GamepadDPadLeft") || input_just_pressed("ArrowLeft") {
            state.facing = rotate_facing_left(state.facing);
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
