mod grid;
mod simulation; 

use crate::grid::*;
use crate::simulation::*;
use bevy::utils::hashbrown::HashMap;
use bevy::prelude::*;
use std::sync::Mutex;


use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::time::Fixed;
use bevy_flycam::prelude::*;

#[derive(Resource)]
pub struct DrawState(bool);

#[derive(Resource)]
pub struct SpawnState(bool);

#[derive(Resource)]
pub struct SpawnBuffer(Mutex<Vec<Particle>>);

#[derive(Resource)]
pub struct DespawnBuffer(Mutex<Vec<Entity>>);


fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(PlayerPlugin)
        .insert_resource(Grid(HashMap::new()))
        .insert_resource(DrawState(false))
        .insert_resource(SpawnState(true))
        .insert_resource(Time::<Fixed>::from_seconds(0.05))
        .insert_resource(SpawnBuffer(Mutex::new(vec![])))
        .insert_resource(DespawnBuffer(Mutex::new(vec![])))
        .add_systems(Startup, (initialize, clear_grid, p2g1, p2g2, update_grid, g2p))
        .add_systems(FixedUpdate, (clear_grid, p2g1, p2g2, update_grid, g2p, spawn).chain())
        .add_systems(Update, (draw, toggle_systems).chain())
        .run();
}

fn toggle_systems(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut draw_state: ResMut<DrawState>,
    mut spawn_state: ResMut<SpawnState>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        draw_state.0 = !draw_state.0;
    }
    if keyboard_input.just_pressed(KeyCode::KeyO) {
        spawn_state.0 = !spawn_state.0;
        
    }
}

// TODO:
// Fix G2P Boundary Conditions for Particles
// See if storing results in temporary array and then locking each mutex once is effective
// (Create hashmap with what chunk each node belongs to and then just go down each list and only
// lock 1 mutex at a time) a flat array could also probably work that just converts between (0..=2,
// ....) to flat coordinates or something similiar
