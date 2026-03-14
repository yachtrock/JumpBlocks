mod camera;
mod debug_ui;
mod edge_detection;
mod layers;
mod native_gamepad;
mod player;
mod world;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dPlugin;

use player::ControlScheme;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "JumpBlocks".to_string(),
                        ..default()
                    }),
                    ..default()
                })
                .build()
                .disable::<bevy::gilrs::GilrsPlugin>(),
            PhysicsPlugins::default()
                .set(PhysicsInterpolationPlugin::interpolate_all()),
            TnuaControllerPlugin::<ControlScheme>::new(PhysicsSchedule),
            TnuaAvian3dPlugin::new(PhysicsSchedule),
        ))
        .add_plugins((
            jumpblocks_voxel::VoxelPlugin,
            world::WorldPlugin,
            player::PlayerPlugin,
            camera::CameraPlugin,
            debug_ui::DebugUiPlugin,
            edge_detection::EdgeDetectionPlugin,
            native_gamepad::NativeGamepadPlugin,
        ))
        .run();
}
