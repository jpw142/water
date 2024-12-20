use rayon::prelude::*;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::collections::HashMap;
use bevy::prelude::*;

/// Langrangian Particle
#[derive(Component)]
struct Particle {
    x: Vec3,    // Position
    v: Vec3,    // Velocity
    c: Mat3,    // Affine Momentum Matrix
    m: f32,     // Mass
}

/// Eulerian Grid Node
struct Node {
    v: Vec3,    // Velocity
    m: f32,     // Mass
    n: u32,     // Material Number
}
impl Node {
    fn zero(&mut self) {
        self.n = 0;
        self.m = 0.;
        self.v = Vec3::ZERO;
    }
}

struct Chunk ( [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE] );

impl Chunk {
    const CHUNK_SIZE: usize = 8;
}
impl Deref for Chunk {
    type Target = [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE];
    fn deref(&self) -> &Self::Target { &self.0 } 
}
impl DerefMut for Chunk { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }  }

// Eulerian Grid Container 
#[derive(Resource)]
struct Grid (HashMap<(i32, i32, i32), Mutex<Chunk>>);
impl Deref for Grid {
    type Target = HashMap<(i32, i32, i32), Mutex<Chunk>>;
    fn deref(&self) -> &Self::Target { &self.0 } 
}
impl DerefMut for Grid { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 } }

/// Sets all nodes in the grid to 0
fn clear_grid ( mut grid: ResMut<Grid> ) {
    grid.par_iter_mut().for_each(|(_, c)| {
        let mut chunk = c.lock().unwrap();
        chunk.iter_mut().for_each(|node| {
            node.zero();
        });
    });
}

// First Transfer of Langrangian Particle Information to Eulerian Grid
fn p2g1 (
    grid: Res<Grid>,
    particles: Query<(&Particle)>
) {
    particles.par_iter().for_each(|p| {
        let n = p.x.trunc(); // Truncates decimal part of float, leaving integer part
        let dx = p.x.fract() - 0.5; // How far away the particle is away from the node
        
        let w = [
            0.5 * (0.5 - dx).powf(2.),
            0.75 - (dx).powf(2.),
            0.5 * (0.5 + dx).powf(2.),
        ];

        let mut m_results = [0.0; 27];
        let mut v_results = [Vec3::ZERO; 27];

        // Calculate contributions of langrangian particle to euclidean grid
        for i in 0..3 { 
            for j in 0..3 {
                for k in 0..3 {
                    let weight = w[i].x * w[j].y * w[k].z; // Quadratic B-Spline  
                    let curr_x = Vec3::from([
                        n.x + (i as f32) - 1.,
                        n.y + (j as f32) - 1.,
                        n.z + (k as f32) - 1.,
                    ]);
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                    let q = p.c * curr_dx;

                    let mass_contrib = weight * p.m;
                    let index = (curr_x.x as usize * 9) + (curr_x.y as usize * 3) + curr_x.z as usize;
                    m_results[index] += mass_contrib;
                    v_results[index] += mass_contrib * (p.v + Vec3::from(q));
                }
            }
        }

        // Lock chunks necessary and load results into it
    })
}

fn main() {
    //App::new().run();
    println!("{}", size_of::<IVec3>());
    println!("{}", size_of::<i32>());
}
