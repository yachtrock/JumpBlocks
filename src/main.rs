mod camera;
mod debug_ui;
mod player;
mod world;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_third_person_camera::ThirdPersonCameraPlugin;
use bevy_tnua::prelude::*;
use bevy_tnua_avian3d::TnuaAvian3dPlugin;

use player::ControlScheme;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "JumpBlocks".to_string(),
                    ..default()
                }),
                ..default()
            }),
            PhysicsPlugins::default(),
            TnuaControllerPlugin::<ControlScheme>::new(PhysicsSchedule),
            TnuaAvian3dPlugin::new(PhysicsSchedule),
            ThirdPersonCameraPlugin,
        ))
        .add_plugins((
            world::WorldPlugin,
            player::PlayerPlugin,
            camera::CameraPlugin,
            debug_ui::DebugUiPlugin,
        ))
        .run();
}
