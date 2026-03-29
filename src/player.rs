use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_tnua::builtins::{TnuaBuiltinJumpConfig, TnuaBuiltinWalkConfig};

use crate::action_state::{ActionState, ConsumedInputs};
use crate::edge_detection::{EdgeDetectionSettings, PrecariousEdge};
use crate::network::LocalPlayer;
use crate::player_state::PlayerState;
use crate::UiInputBlock;
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dSensorShape;
use jumpblocks_voxel::streaming::StreamingAnchor;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(
                Update,
                (player_input, player_lean)
                    .run_if(|block: Res<UiInputBlock>| !block.0),
            );
    }
}

/// Headless player plugin: spawns a player with physics but no visuals.
pub struct HeadlessPlayerPlugin;

impl Plugin for HeadlessPlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_headless_player);
    }
}

#[derive(Component)]
pub struct Player;

/// Marker for the player's visual mesh (child of the physics body).
#[derive(Component)]
pub struct PlayerVisual;

#[derive(Component)]
pub struct PlayerSettings {
    pub run_multiplier: f32,
}

#[derive(Component)]
pub struct LeanSettings {
    /// Max lean angle in degrees when moving at full speed.
    pub max_angle: f32,
    /// Max lean angle in degrees from turning.
    pub turn_max_angle: f32,
    /// How fast the lean interpolates (higher = snappier).
    pub lerp_speed: f32,
}

#[derive(Component, Default)]
pub struct LeanState {
    pub current_direction: Vec3,
    pub current_facing: Quat,
    pub previous_facing_dir: Vec3,
}

#[derive(TnuaScheme)]
#[scheme(basis = TnuaBuiltinWalk)]
pub enum ControlScheme {
    Jump(TnuaBuiltinJump),
}

fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut configs: ResMut<Assets<ControlSchemeConfig>>,
) {
    let player_height = 1.0;
    let player_radius = 0.35;

    let config_handle = configs.add(ControlSchemeConfig {
        basis: TnuaBuiltinWalkConfig {
            float_height: 1.5,
            spring_strength: 1600.0,
            cling_distance: 0.5,
            speed: 10.0,
            acceleration: 20.0,
            air_acceleration: 15.0,
            coyote_time: 0.12,
            ..default()
        },
        jump: TnuaBuiltinJumpConfig {
            height: 3.0,
            ..default()
        },
    });

    let body_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.4, 0.9),
        ..default()
    });
    let visor_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.2, 0.1),
        ..default()
    });

    let mut player = commands.spawn((
        Player,
        LocalPlayer,
        StreamingAnchor,
        PlayerState::default(),
        ActionState::default(),
        ConsumedInputs::default(),
        PlayerSettings { run_multiplier: 1.8 },
        LeanSettings {
            max_angle: 15.0,
            turn_max_angle: 10.0,
            lerp_speed: 8.0,
        },
        LeanState::default(),
        EdgeDetectionSettings::default(),
        PrecariousEdge::default(),
        Transform::from_xyz(0.0, 10.0, 0.0),
        Visibility::default(),
    ));
    player.insert((
        // Physics
        RigidBody::Dynamic,
        Collider::capsule(player_radius, player_height),
        LockedAxes::ROTATION_LOCKED,
        // Tnua character controller
        TnuaController::<ControlScheme>::default(),
        TnuaConfig::<ControlScheme>(config_handle),
        TnuaAvian3dSensorShape(Collider::cylinder(player_radius - 0.01, 0.0)),
    ));
    player.with_children(|parent| {
            // Visual pivot — rotates for facing + lean
            parent
                .spawn((PlayerVisual, Transform::default(), Visibility::default()))
                .with_children(|pivot| {
                    // Body capsule
                    pivot.spawn((
                        Mesh3d(meshes.add(Capsule3d::new(player_radius, player_height))),
                        MeshMaterial3d(body_material),
                    ));
                    // Facing indicator (visor)
                    pivot.spawn((
                        Mesh3d(meshes.add(Sphere::new(0.12))),
                        MeshMaterial3d(visor_material),
                        Transform::from_xyz(0.0, 0.2, -player_radius * 0.85),
                    ));
                });
        });
}

fn spawn_headless_player(
    mut commands: Commands,
    mut configs: ResMut<Assets<ControlSchemeConfig>>,
) {
    let player_radius = 0.35;
    let player_height = 1.0;

    let config_handle = configs.add(ControlSchemeConfig {
        basis: TnuaBuiltinWalkConfig {
            float_height: 1.0,
            speed: 10.0,
            acceleration: 20.0,
            air_acceleration: 15.0,
            coyote_time: 0.12,
            ..default()
        },
        jump: TnuaBuiltinJumpConfig {
            height: 3.0,
            ..default()
        },
    });

    commands.spawn((
        Player,
        LocalPlayer,
        PlayerState::default(),
        ActionState::default(),
        ConsumedInputs::default(),
        PlayerSettings { run_multiplier: 1.8 },
        LeanSettings {
            max_angle: 15.0,
            turn_max_angle: 10.0,
            lerp_speed: 8.0,
        },
        LeanState::default(),
        Transform::from_xyz(0.0, 10.0, 0.0),
        RigidBody::Dynamic,
        Collider::capsule(player_radius, player_height),
        LockedAxes::ROTATION_LOCKED,
        TnuaController::<ControlScheme>::default(),
        TnuaConfig::<ControlScheme>(config_handle),
        TnuaAvian3dSensorShape(Collider::cylinder(player_radius - 0.01, 0.0)),
    ));
    info!("Spawned headless player");
}

pub fn player_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut player_query: Query<
        (&mut TnuaController<ControlScheme>, &PlayerSettings, &mut LeanState, &ConsumedInputs),
        With<Player>,
    >,
    camera_query: Query<&Transform, (With<Camera3d>, Without<Player>)>,
) {
    let Ok((mut controller, settings, mut lean_state, consumed)) = player_query.single_mut() else {
        return;
    };
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    // Gather input from keyboard (respecting consumed inputs from action states)
    let mut input_dir = Vec2::ZERO;
    if (!consumed.0.contains("KeyW") && keyboard.pressed(KeyCode::KeyW))
        || (!consumed.0.contains("ArrowUp") && keyboard.pressed(KeyCode::ArrowUp))
    {
        input_dir.y += 1.0;
    }
    if (!consumed.0.contains("KeyS") && keyboard.pressed(KeyCode::KeyS))
        || (!consumed.0.contains("ArrowDown") && keyboard.pressed(KeyCode::ArrowDown))
    {
        input_dir.y -= 1.0;
    }
    if (!consumed.0.contains("KeyA") && keyboard.pressed(KeyCode::KeyA))
        || (!consumed.0.contains("ArrowLeft") && keyboard.pressed(KeyCode::ArrowLeft))
    {
        input_dir.x -= 1.0;
    }
    if (!consumed.0.contains("KeyD") && keyboard.pressed(KeyCode::KeyD))
        || (!consumed.0.contains("ArrowRight") && keyboard.pressed(KeyCode::ArrowRight))
    {
        input_dir.x += 1.0;
    }

    let mut jump_pressed = keyboard.pressed(KeyCode::Space);
    let mut run_pressed = keyboard.pressed(KeyCode::ShiftLeft)
        || keyboard.pressed(KeyCode::ShiftRight);

    // Gather input from gamepad (respecting consumed inputs from action states)
    for gamepad in gamepads.iter() {
        let stick = Vec2::new(
            gamepad.get(GamepadAxis::LeftStickX).unwrap_or(0.0),
            gamepad.get(GamepadAxis::LeftStickY).unwrap_or(0.0),
        );
        // Apply deadzone
        if stick.length() > 0.15 {
            input_dir += stick;
        }
        if !consumed.0.contains("GamepadSouth") && gamepad.pressed(GamepadButton::South) {
            jump_pressed = true;
        }
        if !consumed.0.contains("GamepadEast") && gamepad.pressed(GamepadButton::East) {
            run_pressed = true;
        }
    }

    // Clamp and normalize input
    if input_dir.length() > 1.0 {
        input_dir = input_dir.normalize();
    }

    // Transform input relative to camera facing direction
    let camera_forward = camera_transform.forward().as_vec3();
    let camera_forward = Vec3::new(camera_forward.x, 0.0, camera_forward.z).normalize_or_zero();
    let camera_right = camera_forward.cross(Vec3::Y);

    let move_dir = camera_forward * input_dir.y + camera_right * input_dir.x;

    // Apply movement via Tnua (speed is controlled by TnuaBuiltinWalkConfig::speed)
    let speed_multiplier = if run_pressed { settings.run_multiplier } else { 1.0 };
    lean_state.current_direction = move_dir * speed_multiplier;
    controller.basis = TnuaBuiltinWalk {
        desired_motion: move_dir * speed_multiplier,
        desired_forward: None,
    };

    controller.initiate_action_feeding();
    if jump_pressed {
        controller.action(ControlScheme::Jump(TnuaBuiltinJump::default()));
    }
}

fn player_lean(
    time: Res<Time>,
    mut player_query: Query<(&LeanSettings, &mut LeanState), With<Player>>,
    mut visual_query: Query<&mut Transform, (With<PlayerVisual>, Without<Player>)>,
) {
    let Ok((lean_settings, mut lean_state)) = player_query.single_mut() else {
        return;
    };
    let Ok(mut visual_transform) = visual_query.single_mut() else {
        return;
    };

    let dt = time.delta_secs();
    let move_dir = lean_state.current_direction;

    // Update facing direction when moving
    if move_dir.length_squared() > 0.01 {
        let target_facing = Quat::from_rotation_arc(Vec3::NEG_Z, move_dir.normalize());
        let facing_lerp = (lean_settings.lerp_speed * dt).min(1.0);
        lean_state.current_facing = lean_state.current_facing.slerp(target_facing, facing_lerp);
    }

    // Get the current facing flat direction for lean calculations
    let facing_flat = (lean_state.current_facing * Vec3::NEG_Z).normalize_or_zero();

    // Compute turn rate (change in facing direction)
    let turn_cross = lean_state.previous_facing_dir.cross(facing_flat);
    let turn_amount = turn_cross.y.clamp(-1.0, 1.0);
    lean_state.previous_facing_dir = facing_flat;

    // Forward lean from movement speed
    let forward_amount = move_dir.dot(facing_flat);
    let forward_lean_rad = (-forward_amount * lean_settings.max_angle).to_radians();
    // Side lean from turning
    let turn_lean_rad = (-turn_amount * lean_settings.turn_max_angle).to_radians();

    // Combine facing yaw with lean (pitch + roll)
    let lean = Quat::from_rotation_x(forward_lean_rad) * Quat::from_rotation_z(turn_lean_rad);
    let target_rotation = lean_state.current_facing * lean;

    // Apply smoothly
    let lerp_t = (lean_settings.lerp_speed * dt).min(1.0);
    visual_transform.rotation = visual_transform.rotation.slerp(target_rotation, lerp_t);
}

