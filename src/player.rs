use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_third_person_camera::ThirdPersonCameraTarget;
use bevy_tnua::builtins::{TnuaBuiltinJumpConfig, TnuaBuiltinWalkConfig};
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dSensorShape;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, player_input);
    }
}

#[derive(Component)]
pub struct Player;

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
            float_height: 1.0,
            acceleration: 40.0,
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
        Mesh3d(meshes.add(Capsule3d::new(player_radius, player_height))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.4, 0.9),
            ..default()
        })),
        Transform::from_xyz(0.0, 2.0, 0.0),
        // Physics
        RigidBody::Dynamic,
        Collider::capsule(player_radius, player_height),
        LockedAxes::ROTATION_LOCKED,
        // Tnua character controller
        TnuaController::<ControlScheme>::default(),
        TnuaConfig::<ControlScheme>(config_handle),
        TnuaAvian3dSensorShape(Collider::cylinder(player_radius - 0.01, 0.0)),
        // Camera target
        ThirdPersonCameraTarget,
    ));
}

fn player_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut player_query: Query<&mut TnuaController<ControlScheme>, With<Player>>,
    camera_query: Query<&Transform, (With<Camera3d>, Without<Player>)>,
) {
    let Ok(mut controller) = player_query.single_mut() else {
        return;
    };
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };

    // Gather input from keyboard
    let mut input_dir = Vec2::ZERO;
    if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        input_dir.y += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        input_dir.y -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        input_dir.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        input_dir.x += 1.0;
    }

    let mut jump_pressed = keyboard.pressed(KeyCode::Space);

    // Gather input from gamepad
    for gamepad in gamepads.iter() {
        let stick = Vec2::new(
            gamepad.get(GamepadAxis::LeftStickX).unwrap_or(0.0),
            gamepad.get(GamepadAxis::LeftStickY).unwrap_or(0.0),
        );
        // Apply deadzone
        if stick.length() > 0.15 {
            input_dir += stick;
        }
        if gamepad.pressed(GamepadButton::South) {
            jump_pressed = true;
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

    // Apply movement via Tnua
    let speed = 8.0;
    controller.basis = TnuaBuiltinWalk {
        desired_motion: move_dir * speed,
        desired_forward: Dir3::new(move_dir).ok(),
    };

    controller.initiate_action_feeding();
    if jump_pressed {
        controller.action(ControlScheme::Jump(TnuaBuiltinJump::default()));
    }
}
