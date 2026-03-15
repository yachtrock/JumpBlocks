// Building state — entered when the player selects an inventory item.
// B button (GamepadEast / Escape) exits, ZR/F places a block.
// Hold ZL/Q to show options panel with rotate (dpad left/right) and
// change position (dpad up/down).

fn on_enter(ctx) {
    log("Entered building state with item: " + ctx.item_name);
    replicate("block_type", ctx.item_name);
    replicate("slot", ctx.selected_slot);

    // Set button hints for the HUD
    set_button_hint("Place Block", "F", "ZR");
    set_button_hint("Put Away", "Esc", "B");
    set_button_hint("Options", "Q", "ZL");

    #{ item_name: ctx.item_name, selected_slot: ctx.selected_slot, options_held: false }
}

fn on_update(state, input) {
    // Consume inputs so default player bindings don't fire
    consume_input("GamepadEast");
    consume_input("GamepadRightTrigger");
    consume_input("KeyF");
    consume_input("KeyQ");
    consume_input("GamepadLeftTrigger");

    // Options panel: hold ZL / Q
    let options_held = input_pressed("GamepadLeftTrigger") || input_pressed("KeyQ");

    // Exit on B / Escape
    if input_just_pressed("GamepadEast") || input_just_pressed("Escape") {
        exit_state();
    }

    // Place block on ZR / F
    if input_just_pressed("GamepadRightTrigger") || input_just_pressed("KeyF") {
        log("Place block! (item: " + state.item_name + ")");
        emit("place_block");
    }

    if options_held {
        // Consume dpad/arrow keys so they don't move the character
        consume_input("GamepadDPadLeft");
        consume_input("GamepadDPadRight");
        consume_input("GamepadDPadUp");
        consume_input("GamepadDPadDown");
        consume_input("ArrowLeft");
        consume_input("ArrowRight");
        consume_input("ArrowUp");
        consume_input("ArrowDown");

        // Show options panel hints instead of normal hints
        clear_button_hints();
        set_button_hint_pair("Rotate", "←", "→", "D←", "D→");
        set_button_hint_pair("Position", "↑", "↓", "D↑", "D↓");

        // Handle rotate
        if input_just_pressed("GamepadDPadLeft") || input_just_pressed("ArrowLeft") {
            log("Rotate left (item: " + state.item_name + ")");
            emit("rotate_left");
        }
        if input_just_pressed("GamepadDPadRight") || input_just_pressed("ArrowRight") {
            log("Rotate right (item: " + state.item_name + ")");
            emit("rotate_right");
        }

        // Handle position change
        if input_just_pressed("GamepadDPadUp") || input_just_pressed("ArrowUp") {
            log("Position up (item: " + state.item_name + ")");
            emit("position_up");
        }
        if input_just_pressed("GamepadDPadDown") || input_just_pressed("ArrowDown") {
            log("Position down (item: " + state.item_name + ")");
            emit("position_down");
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

    state.options_held = options_held;
    state
}

fn on_exit(state) {
    clear_button_hints();
    log("Exited building state");
}
