mod grid;
mod simulation; 

use crate::grid::*;
use crate::simulation::*;

use std::collections::HashMap;
use bevy::prelude::*;


use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::time::Fixed;
use bevy_flycam::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(PlayerPlugin)
        .insert_resource(Grid(HashMap::new()))
        .add_systems(Startup, initialize)
        .insert_resource(Time::<Fixed>::from_seconds(0.05))
        .add_systems(FixedUpdate, (clear_grid, p2g1, p2g2, update_grid, g2p).chain())
        .add_systems(Update, draw)
        .run();
}


// TODO:
// Fix G2P Boundary Conditions for Particles
// See if storing results in temporary array and then locking each mutex once is effective
// (Create hashmap with what chunk each node belongs to and then just go down each list and only
// lock 1 mutex at a time) a flat array could also probably work that just converts between (0..=2,
// ....) to flat coordinates or something similiar
