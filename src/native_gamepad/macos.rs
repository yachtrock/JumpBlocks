use bevy::ecs::message::MessageWriter;
use bevy::input::gamepad::{
    GamepadAxis, GamepadConnection, GamepadConnectionEvent, RawGamepadAxisChangedEvent,
    RawGamepadButtonChangedEvent, RawGamepadEvent,
};
use bevy::prelude::*;
use objc2_game_controller::{GCController, GCDevice, GCExtendedGamepad};

use super::{ButtonIndex, ControllerState, NativeGamepads, BUTTON_COUNT};

/// Minimum change in value to emit an event (avoids noise).
const AXIS_EPSILON: f32 = 0.0001;

pub fn startup() {
    unsafe {
        GCController::setShouldMonitorBackgroundEvents(true);
        GCController::startWirelessControllerDiscoveryWithCompletionHandler(None);
    }
    info!("Native gamepad: Apple Game Controller framework initialized");
}

pub fn poll_gamepads(
    mut commands: Commands,
    mut native_gamepads: ResMut<NativeGamepads>,
    mut events: MessageWriter<RawGamepadEvent>,
    mut connection_events: MessageWriter<GamepadConnectionEvent>,
    mut button_events: MessageWriter<RawGamepadButtonChangedEvent>,
    mut axis_events: MessageWriter<RawGamepadAxisChangedEvent>,
) {
    let controllers = unsafe { GCController::controllers() };

    // Build set of currently-connected controller IDs
    let mut current_ids: Vec<usize> = Vec::new();

    for controller in controllers.iter() {
        let controller_id = objc2::rc::Retained::as_ptr(&controller) as usize;
        current_ids.push(controller_id);

        let Some(gamepad) = (unsafe { controller.extendedGamepad() }) else {
            continue;
        };

        // Check if this is a new controller
        if !native_gamepads.controllers.contains_key(&controller_id) {
            let entity = commands.spawn_empty().id();
            let name = unsafe {
                controller
                    .vendorName()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "Unknown Controller".into())
            };

            // Nintendo controllers have swapped face button labels vs position
            let nintendo_layout = matches!(
                name.as_str(),
                "Pro Controller" | "Joy-Con (L/R)" | "Joy-Con (L)" | "Joy-Con (R)"
            );
            if nintendo_layout {
                info!("Native gamepad connected: {name} (entity {entity}) [Nintendo layout]");
            } else {
                info!("Native gamepad connected: {name} (entity {entity})");
            }

            let event = GamepadConnectionEvent {
                gamepad: entity,
                connection: GamepadConnection::Connected {
                    name: name.clone(),
                    vendor_id: None,
                    product_id: None,
                },
            };
            events.write(event.clone().into());
            connection_events.write(event);

            native_gamepads.controllers.insert(
                controller_id,
                ControllerState {
                    entity,
                    name,
                    nintendo_layout,
                    prev_axes: [0.0; 4],
                    prev_buttons: [0.0; BUTTON_COUNT],
                },
            );
        }

        // Read current state and emit events for changes
        let state = native_gamepads.controllers.get_mut(&controller_id).unwrap();
        let entity = state.entity;

        let nintendo = state.nintendo_layout;
        poll_axes(&gamepad, state, entity, &mut events, &mut axis_events);
        poll_buttons(
            &gamepad,
            state,
            entity,
            nintendo,
            &mut events,
            &mut button_events,
        );
    }

    // Detect disconnected controllers
    let disconnected: Vec<usize> = native_gamepads
        .controllers
        .keys()
        .filter(|id| !current_ids.contains(id))
        .copied()
        .collect();

    for id in disconnected {
        if let Some(state) = native_gamepads.controllers.remove(&id) {
            info!("Native gamepad disconnected: {} (entity {})", state.name, state.entity);
            let event =
                GamepadConnectionEvent::new(state.entity, GamepadConnection::Disconnected);
            events.write(event.clone().into());
            connection_events.write(event);
        }
    }
}

fn poll_axes(
    gamepad: &GCExtendedGamepad,
    state: &mut ControllerState,
    entity: Entity,
    events: &mut MessageWriter<RawGamepadEvent>,
    axis_events: &mut MessageWriter<RawGamepadAxisChangedEvent>,
) {
    let axes = unsafe {
        [
            (GamepadAxis::LeftStickX, gamepad.leftThumbstick().xAxis().value()),
            (GamepadAxis::LeftStickY, gamepad.leftThumbstick().yAxis().value()),
            (GamepadAxis::RightStickX, gamepad.rightThumbstick().xAxis().value()),
            (GamepadAxis::RightStickY, gamepad.rightThumbstick().yAxis().value()),
        ]
    };

    for (i, (axis, value)) in axes.iter().enumerate() {
        if (value - state.prev_axes[i]).abs() > AXIS_EPSILON {
            state.prev_axes[i] = *value;
            let event = RawGamepadAxisChangedEvent {
                gamepad: entity,
                axis: *axis,
                value: *value,
            };
            events.write(event.clone().into());
            axis_events.write(event);
        }
    }
}

fn poll_buttons(
    gamepad: &GCExtendedGamepad,
    state: &mut ControllerState,
    entity: Entity,
    nintendo_layout: bool,
    events: &mut MessageWriter<RawGamepadEvent>,
    button_events: &mut MessageWriter<RawGamepadButtonChangedEvent>,
) {
    unsafe {
        // Apple GCF uses the button LABEL, not position.
        // Nintendo: A=right, B=bottom, X=top, Y=left
        // Xbox:     A=bottom, B=right, X=left, Y=top
        // Bevy uses positional names: South=bottom, East=right, etc.
        let (south, east, north, west) = if nintendo_layout {
            // Nintendo A (right) → East, B (bottom) → South, X (top) → North, Y (left) → West
            (
                gamepad.buttonB().value(), // B = bottom = South
                gamepad.buttonA().value(), // A = right  = East
                gamepad.buttonX().value(), // X = top    = North
                gamepad.buttonY().value(), // Y = left   = West
            )
        } else {
            // Xbox/generic: A=bottom, B=right, X=left, Y=top
            (
                gamepad.buttonA().value(),
                gamepad.buttonB().value(),
                gamepad.buttonY().value(),
                gamepad.buttonX().value(),
            )
        };

        let mut buttons: [(ButtonIndex, f32); BUTTON_COUNT] = [
            (ButtonIndex::South, south),
            (ButtonIndex::East, east),
            (ButtonIndex::West, west),
            (ButtonIndex::North, north),
            (ButtonIndex::LeftTrigger, gamepad.leftShoulder().value()),
            (ButtonIndex::LeftTrigger2, gamepad.leftTrigger().value()),
            (ButtonIndex::RightTrigger, gamepad.rightShoulder().value()),
            (ButtonIndex::RightTrigger2, gamepad.rightTrigger().value()),
            (ButtonIndex::Start, gamepad.buttonMenu().value()),
            // Optional buttons — default to 0 if absent
            (ButtonIndex::Select, 0.0),
            (ButtonIndex::LeftThumb, 0.0),
            (ButtonIndex::RightThumb, 0.0),
            // D-pad
            (ButtonIndex::DPadUp, gamepad.dpad().up().value()),
            (ButtonIndex::DPadDown, gamepad.dpad().down().value()),
            (ButtonIndex::DPadLeft, gamepad.dpad().left().value()),
            (ButtonIndex::DPadRight, gamepad.dpad().right().value()),
            (ButtonIndex::Mode, 0.0),
            (ButtonIndex::C, 0.0),
        ];

        // Fill in optional buttons
        if let Some(options) = gamepad.buttonOptions() {
            buttons[ButtonIndex::Select as usize].1 = options.value();
        }
        if let Some(left_thumb) = gamepad.leftThumbstickButton() {
            buttons[ButtonIndex::LeftThumb as usize].1 = left_thumb.value();
        }
        if let Some(right_thumb) = gamepad.rightThumbstickButton() {
            buttons[ButtonIndex::RightThumb as usize].1 = right_thumb.value();
        }
        if let Some(home) = gamepad.buttonHome() {
            buttons[ButtonIndex::Mode as usize].1 = home.value();
        }

        for (idx, value) in &buttons {
            let i = *idx as usize;
            if (value - state.prev_buttons[i]).abs() > AXIS_EPSILON {
                state.prev_buttons[i] = *value;
                let event = RawGamepadButtonChangedEvent {
                    gamepad: entity,
                    button: idx.to_gamepad_button(),
                    value: *value,
                };
                events.write(event.clone().into());
                button_events.write(event);
            }
        }
    }
}
