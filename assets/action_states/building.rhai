// Building state — entered when the player selects an inventory item.
// B button (GamepadEast / Escape) exits, ZR (GamepadRightTrigger) places a block.

fn on_enter(ctx) {
    log("Entered building state with item: " + ctx.item_name);
    replicate("block_type", ctx.item_name);
    replicate("slot", ctx.selected_slot);

    // Set button hints for the HUD
    set_button_hint("Place Block", "F", "ZR");
    set_button_hint("Put Away", "Esc", "B");
    set_button_hint("Options", "Q", "ZL");

    #{ item_name: ctx.item_name, selected_slot: ctx.selected_slot }
}

fn on_update(state, input) {
    // Consume inputs so default player bindings don't fire
    consume_input("GamepadEast");
    consume_input("GamepadRightTrigger");

    if input_just_pressed("GamepadEast") || input_just_pressed("Escape") {
        exit_state();
    }

    if input_just_pressed("GamepadRightTrigger") {
        log("Place block! (item: " + state.item_name + ")");
        emit("place_block");
    }

    state
}

fn on_exit(state) {
    clear_button_hints();
    log("Exited building state");
}
