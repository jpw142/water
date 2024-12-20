//use rayon::prelude::*;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use bevy::prelude::*;

/// Langrangian Particle
struct Particle {
    //x: Vec3,    // Position
    //v: Vec3,    // Velocity
    //C: Mat3,    // Affine Momentum Matrix
    m: f32,     // Mass
}

struct Chunk {
    chunk: [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE],
}

impl Chunk {
    const CHUNK_SIZE: usize = 8;
}
impl Deref for Chunk {
    type Target = [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE];
    fn deref(&self) -> &Self::Target { &self.chunk } 
}

/// Eulerian Grid Node
struct Node {
    v: Vec3,    // Velocity
    m: f32,     // Mass
}

// Eulerian Grid Container
struct Grid {
    grid: HashMap<IVec3, Mutex<i32>>
}

// Allows you to Dereference inner list without typing p.p or list.p 
impl Deref for Grid {
    type Target = HashMap<IVec3, Mutex<i32>>;
    fn deref(&self) -> &Self::Target { &self.grid }
}

fn main() {
    //App::new().run();
    println!("{}", size_of::<Node>());
}
