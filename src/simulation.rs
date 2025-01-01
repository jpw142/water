#![allow(non_upper_case_globals)]

use rayon::prelude::*;
use crate::DrawState;
use bevy::{prelude::*, gizmos::gizmos, color::palettes::{tailwind::{RED_500, BLUE_500}, css::GHOST_WHITE}};
use std::collections::HashMap;

use crate::grid::*;

const dt: f32 = 0.2;
const gravity: f32 = -0.3;
const rest_density: f32 = 4.0;
const dynamic_viscosity: f32 = 0.1;
const eos_stiffness: f32 = 10.0;
const eos_power: f32 = 4.;


/// Langrangian Particle
#[derive(Component)]
pub struct Particle {
    x: Vec3,    // Position
    v: Vec3,    // Velocity
    c: Mat3,    // Affine Momentum Matrix
    m: f32,     // Mass
    p: u16    // material prime index
}

/// Eulerian Grid Node
#[derive(Copy, Clone)]
pub struct Node {
    v: Vec3,    // Velocity
    m: f32,     // Mass
    p: u32,     // Material Number
}

impl Default for Node {
    fn default() -> Self {
        Node { v: Vec3::ZERO, m: 0., p: 0}
    }
}

impl Node {
    fn zero(&mut self) {
        self.p = 0;
        self.m = 0.0;
        self.v = Vec3::ZERO;
    }
}


/// Sets all nodes in the grid to 0
pub fn clear_grid ( mut grid: ResMut<Grid> ) {
    grid.par_iter_mut().for_each(|(_, c)| {
        let mut chunk = c.lock().expect("Error locking Mutex for clear_grid");
        chunk.iter_mut().for_each(|node| {
            node.zero();
        });
    });
}

pub fn p2g1 (
    grid: ResMut<Grid>,
    query: Query<&Particle>,
    ) {
    //.par_iter_mut()
    query.par_iter().for_each(|p| {
        let n = p.x.trunc(); // Truncates decimal part of float, leaving integer part
        let dx = p.x.fract() - 0.5; // How far away the particle is away from the node

        let w = [
            0.5 * (0.5 - dx).powf(2.),
            0.75 - (dx).powf(2.),
            0.5 * (0.5 + dx).powf(2.),
        ];


        // Buffer to store new node informatio nin to avoid locking mutex 27 times
        let mut node_buffer: HashMap<IVec3, Vec<(usize, f32, Vec3)>> = HashMap::new();

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
                    let icurr_x = curr_x.as_ivec3();
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n


                    // Pre-calculate values we will send to mutex
                    let q = p.c * curr_dx;
                    let mass_contrib = weight * p.m;
                    let v_contrib = mass_contrib * (p.v + Vec3::from(q));

                    // Get Mutex
                    let chunk_pos = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let node_index = Chunk::node_world_pos_to_index(icurr_x);

                    // Append nodes to node buffer
                    node_buffer.entry(chunk_pos).or_insert(vec![]).push((node_index, mass_contrib, v_contrib));
                    
                }
            }
        }

        // Lock the mutexes and write the Nod data
        for (chunk, nodes) in node_buffer {
            let mut chunk = grid.get(&chunk).expect("Particle out of bounds p2g1").lock().expect("Error locking mutex for p2g1");
            for node in nodes {
                chunk[node.0].m += node.1;
                chunk[node.0].v += node.2;
            }
            std::mem::drop(chunk); // Unlock Mutex
        }
    });
}

pub fn p2g2 (
    grid: ResMut<Grid>,
    query: Query<&Particle>,
    ) {
    query.par_iter().for_each(|p| {
        let n = p.x.trunc(); // Truncates decimal part of float, leaving integer part
        let dx = p.x.fract() - 0.5; // How far away the particle is away from the node

        let w = [
            0.5 * (0.5 - dx).powf(2.),
            0.75 - (dx).powf(2.),
            0.5 * (0.5 + dx).powf(2.),
        ];


        let mut density: f32 = 0.;
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
                    let icurr_x = curr_x.as_ivec3();

                    // Get Mutex
                    let chunk_index = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let n_index = Chunk::node_world_pos_to_index(icurr_x);
                    // TODO: Could be improved by pre-fetching masses
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock().expect("Error locking mutex for p2g");

                    // Do work inside mutex
                    density += chunk[n_index].m * weight;

                    std::mem::drop(chunk); // Unlock Mutex
                }
            }
        }

        let volume = p.m / density;
        let pressure = (-0.1_f32).max(eos_stiffness * (density / rest_density).powf(eos_power) - 1.);
        // Identity Matrix
        let mut stress = Mat3::from_cols_array(&[
                                               -pressure, 0., 0., 
                                               0., -pressure, 0.,
                                               0., 0., -pressure,
        ]);
        let mut strain = p.c;

        let trace = strain.x_axis.x + strain.y_axis.y + strain.z_axis.z;
        strain.x_axis.x = trace;
        strain.y_axis.y = trace;
        strain.z_axis.z = trace;

        let viscosity_term = dynamic_viscosity * strain;
        stress += viscosity_term;

        let eq_16_term_0 = -volume * 4. * stress * dt;

        for i in 0..3 { 
            for j in 0..3 {
                for k in 0..3 {
                    let weight = w[i].x * w[j].y * w[k].z; // Quadratic B-Spline  
                    let curr_x = Vec3::from([
                                            n.x + (i as f32) - 1.,
                                            n.y + (j as f32) - 1.,
                                            n.z + (k as f32) - 1.,
                    ]);
                    let icurr_x = curr_x.as_ivec3();
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                                                        
                    let momentum = Vec3::from(eq_16_term_0 * weight * curr_dx);

                    // Get Mutex
                    let chunk_index = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let n_index = Chunk::node_world_pos_to_index(icurr_x);
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock().expect("Error locking mutex for p2g");


                    // Do work inside mutex
                    chunk[n_index].v += momentum;

                    std::mem::drop(chunk); // Unlock Mutex
                }
            }
        }
    });
}

//
//  Edge mask formatted like a dice
//      +--------+
//     /        /|
//    /    6   / |
//   +--------+  |
//   |        | 4|
//   |        |  +
//   |    5   | /
//   |        |/
//   +--------+
// 1 (0, -1, 0)  
// 2 (0, 0, -1)
// 3 (-1, 0, 0)
// 4 (+1, 0, 0)
// 5 (0, 0, +1)
// 6 (0, +1, 0)
//
pub fn update_grid ( grid: Res<Grid>) {

    grid.par_iter().for_each(|(_, c)| {
        let mut chunk = c.lock().expect("Error locking Mutex for update_grid");
        let chunk_edge_mask = chunk.edge_mask;

        chunk.iter_mut().enumerate().for_each(|(i, node)| {
            // Boundary condition
            let nlp = Chunk::index_to_node_local_pos(i);


            if node.m > 0. {
                node.v /= node.m;
                node.v.y += dt * gravity;
            }

            if chunk_edge_mask != 0 {
                let node_edge_mask = Chunk::node_local_pos_to_edge_mask(nlp);
                if node_edge_mask & chunk_edge_mask != 0 {node.v = Vec3::ZERO};
            }
        });
    });
}

pub fn g2p (
    grid: Res<Grid>,
    mut particles: Query<&mut Particle>
) {
   particles.par_iter_mut().for_each(|mut p| {
        p.v = Vec3::ZERO;

        let n = p.x.trunc(); // Truncates decimal part of float, leaving integer part
        let dx: Vec3 = p.x.fract() - 0.5; // How far away the particle is away from the node
        
        let w: [Vec3; 3]= [
            0.5 * (0.5 - dx).powf(2.),
            0.75 - (dx).powf(2.),
            0.5 * (0.5 + dx).powf(2.),
        ];

        let mut b = Mat3::ZERO;
        let mut edge_mask = 0; 
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let weight = w[i].x * w[j].y * w[k].z; // Quadratic B-Spline  
                    let curr_x = Vec3::from([
                        n.x + (i as f32) - 1.,
                        n.y + (j as f32) - 1.,
                        n.z + (k as f32) - 1.,
                    ]);
                    let icurr_x = curr_x.as_ivec3();
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                    // Get Mutex
                    let chunk_index = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let n_index = Chunk::node_world_pos_to_index(icurr_x);
                    let chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock().expect("Error locking mutex for p2g");

                    // Do work inside mutex
                    let mut w_v = chunk[n_index].v;
                    
                    // Get the edge_mask for free while locking the first time
                    if i == 0 && j == 0 && k == 0 {
                        edge_mask = chunk.edge_mask;
                    }

                    std::mem::drop(chunk); // Unlock Mutex

                    w_v *= weight;
                    let term = Mat3::from_cols(w_v * curr_dx.x, w_v * curr_dx.y, w_v * curr_dx.z);
                    b += term;
                    p.v += w_v;
                }
            }
        }
        p.c = b.mul_scalar(4.);

        let end = (16 - (Chunk::EDGE_BUFFER_SIZE)) as f32;

        let local_pos = Chunk::index_to_node_local_pos(Chunk::node_world_pos_to_index(n.as_ivec3()));

        let x_n = p.x + p.v;

        if x_n.y < 2. {p.v.y += Chunk::EDGE_BUFFER_SIZE as f32 - x_n.y as f32} // 1
        if x_n.z < 2. {p.v.z += Chunk::EDGE_BUFFER_SIZE as f32 - x_n.z as f32} // 2
        if x_n.x < 2. {p.v.x += Chunk::EDGE_BUFFER_SIZE as f32 - x_n.x as f32} // 3
        if x_n.x > end {p.v.x += end - x_n.x as f32} //4 
        if x_n.z > end {p.v.z += end - x_n.z as f32} // 5
        if x_n.y > end {p.v.y += end - x_n.y as f32} // 6

        let v = p.v;
        p.x += v * dt; 
        // println!("{}", p.x);
    })
}

pub fn initialize(
    mut commands: Commands,
    mut grid: ResMut<Grid>,
) {
    for i in 0..4 {
        for j in 0..4 { 
            for k in 0..4 { 
                grid.new_chunk(&IVec3::new(i, j, k));
            }
        }
    }

    for x in 3..13 {
        for y in 3..13{
            for z in 3..13 {
                commands.spawn(Particle {
                    x: Vec3::new(x as f32, y as f32, z as f32),
                    v: Vec3::ZERO,
                    c: Mat3::ZERO,
                    p: 0,
                    m: 1.
                });
            }
        }
    }
}

pub fn spawn(
    mut commands: Commands,
    ) {
    commands.spawn(Particle {
        x: Vec3::new(8., 12., 8.),
        v: Vec3::ZERO,
        c: Mat3::ZERO,
        p: 0, m: 1.
    });
}

pub fn draw(
    particles: Query<&Particle>,
    mut gizmos: Gizmos,
    mut draw_state: ResMut<DrawState>,
) {
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                gizmos.sphere(Vec3::from((i as f32, j as f32, k as f32)) * 4., 0.05, GHOST_WHITE);
            }}}
    gizmos.cuboid( 
        Transform::from_translation(Vec3::from((8., 8., 8.))).with_scale(Vec3::splat(16.)),
        GHOST_WHITE
        );

    if draw_state.0 == false {
        return;
    }

    particles.iter().for_each(|p| {
        gizmos.sphere(p.x, 0.1, BLUE_500);
    });
}
