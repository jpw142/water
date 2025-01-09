#![allow(non_upper_case_globals)]

use rayon::prelude::*;
use crate::{DrawState, SpawnState, SpawnBuffer, DespawnBuffer};
use bevy::{prelude::*, gizmos::gizmos, color::palettes::{tailwind::{RED_500, BLUE_500, ORANGE_500, LIME_500, GRAY_500}, css::GHOST_WHITE}, math::{Vec3A, Mat3A}};
use bevy::utils::hashbrown::HashMap;
use bevy::log::*;

use crate::grid::*;

const dt: f32 = 0.2;
const gravity: f32 = -0.3;
const rest_density: f32 = 4.0;
const dynamic_viscosity: f32 = 0.1;
const eos_stiffness: f32 = 10.0;
const eos_power: f32 = 4.;

const sim_max_pos: usize = 64;


/// Langrangian Particle
#[derive(Component, Clone, Copy)]
pub struct Particle {
    x: Vec3A,    // Position
    v: Vec3A,    // Velocity
    c: Mat3A,    // Affine Momentum Matrix
    m: f32,     // Mass
    p: u16    // material prime index
}

/// Eulerian Grid Node
#[derive(Copy, Clone)]
pub struct Node {
    v: Vec3A,    // Velocity
    m: f32,     // Mass
    p: u32,     // Material Number
}

impl Default for Node {
    fn default() -> Self {
        Node { v: Vec3A::ZERO, m: 0., p: 1}
    }
}

impl Node {
    fn zero(&mut self) {
        self.p = 1;
        self.m = 0.0;
        self.v = Vec3A::ZERO;
    }
}


/// Sets all nodes in the grid to 0
pub fn clear_grid ( mut grid: ResMut<Grid> ) {
    grid.0.iter().par_bridge().for_each(|(_, c)| {
        let mut chunk = c.lock();
        chunk.iter_mut().for_each(|node| {
            node.zero();
        });
    });
}

pub fn p2g1 (
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


        // Buffer to store new node informatio nin to avoid locking mutex 27 times
        //let mut node_buffer: HashMap<IVec3, Vec<(usize, f32, Vec3A)>> = HashMap::new();

        let results : [(IVec3, usize, f32, Vec3A); 27] = core::array::from_fn(|index| {
            let i = index / 9;
            let j = (index % 9) / 3; 
            let k = index % 3;

            let weight = w[i].x * w[j].y * w[k].z;

            let curr_x = n + (Vec3A::new(i as f32, j as f32, k as f32) - Vec3A::ONE);
            let curr_dx = (curr_x - p.x) + 0.5;

            let q = p.c * curr_dx;
            
            // Contribution to the grid based on a quadratic BSpline weighting
            let mass_contribution = weight * p.m;
            let velocity_contribution = mass_contribution * (p.v + Vec3A::from(q));

            let chunk_pos = Chunk::node_world_pos_to_chunk_pos(curr_x.as_ivec3()); 
            let node_index = Chunk::node_world_pos_to_index(curr_x.as_ivec3());

            (chunk_pos, node_index, mass_contribution, velocity_contribution)
        });

        let mut current_chunk = results[0].0;
        let mut chunk_lock = grid.get(&current_chunk).expect("Particle out of bounds p2g1").lock();

        for (chunk, index, mass, velocity) in results {
            if chunk != current_chunk {
                drop(chunk_lock);
                chunk_lock = grid.get(&chunk).expect("Particle out of bounds p2g1").lock();
                current_chunk = chunk;
            }
            chunk_lock[index].m += mass;
            chunk_lock[index].v += velocity;
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
                    let curr_x = Vec3A::from([
                                            n.x + (i as f32) - 1.,
                                            n.y + (j as f32) - 1.,
                                            n.z + (k as f32) - 1.,
                    ]);
                    let icurr_x = curr_x.as_ivec3();

                    // Get Mutex
                    let chunk_index = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let n_index = Chunk::node_world_pos_to_index(icurr_x);
                   // TODO: Could be improved by pre-fetching masses
                    //
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock();

                    // Do work inside mutex
                    density += chunk[n_index].m * weight;

                    std::mem::drop(chunk); // Unlock Mutex
                }
            }
        }

        let volume = p.m / density;
        let pressure = (-0.1_f32).max(eos_stiffness * (density / rest_density).powf(eos_power) - 1.);
        // Identity Matrix
        let mut stress = Mat3A::from_cols_array(&[
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
                    let curr_x = Vec3A::from([
                                            n.x + (i as f32) - 1.,
                                            n.y + (j as f32) - 1.,
                                            n.z + (k as f32) - 1.,
                    ]);
                    let icurr_x = curr_x.as_ivec3();
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                                                        
                    let momentum = Vec3A::from(eq_16_term_0 * weight * curr_dx);

                    // Get Mutex
                    //
                    let chunk_index = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let n_index = Chunk::node_world_pos_to_index(icurr_x);
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock();


                    // Do work inside mutex
                    chunk[n_index].v += momentum;
                    if i == 0 && j == 0 && k == 0 {
                        chunk[n_index].p *= p.p as u32;
                    }

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
pub fn update_grid ( 
    grid: Res<Grid>,
    spawn_buf: Res<SpawnBuffer>,
) {
    grid.0.iter().par_bridge().for_each(|(chunk_pos, c)| {
        let mut chunk = c.lock();
        let chunk_edge_mask = chunk.edge_mask;

        chunk.iter_mut().enumerate().for_each(|(i, node)| {
            // Boundary condition
            let nlp = Chunk::index_to_node_local_pos(i);

            if node.p % 6 == 0 {
                let mut sb = spawn_buf.0.lock().unwrap();
                let x: Vec3A = Vec3A::from((nlp.as_vec3() + (chunk_pos.as_vec3() * Chunk::CHUNK_SIZE as f32)) + Vec3::splat(0.5)) + (node.v.normalize() * -0.25);
                sb.push(Particle{
                    x, 
                    m: 2.,
                    p: 5,
                    c: Mat3A::ZERO,
                    v: Vec3A::ZERO,
                });
                node.p = 6;
            }
            else {
                node.p = 1;
            }
            if node.m > 0. {
                // implementing material combination
                node.v /= node.m;
                node.v.y += dt * gravity;
            }

            if chunk_edge_mask != 0 {
                let node_edge_mask = Chunk::node_local_pos_to_edge_mask(nlp);
                if node_edge_mask & chunk_edge_mask != 0 {node.v = Vec3A::ZERO};
            }
        });
    });
}

pub fn g2p (
    grid: Res<Grid>,
    mut particles: Query<(Entity, &mut Particle)>,
    despawn_buf: Res<DespawnBuffer>,
) {
   particles.par_iter_mut().for_each(|(e, mut p)| {
        p.v = Vec3A::ZERO;

        let n = p.x.trunc(); // Truncates decimal part of float, leaving integer part
        let dx: Vec3A = p.x.fract() - 0.5; // How far away the particle is away from the node
        
        let w: [Vec3A; 3]= [
            0.5 * (0.5 - dx).powf(2.),
            0.75 - (dx).powf(2.),
            0.5 * (0.5 + dx).powf(2.),
        ];

        let mut b = Mat3A::ZERO;
        let mut edge_mask = 0; 
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let weight = w[i].x * w[j].y * w[k].z; // Quadratic B-Spline  
                    let curr_x = Vec3A::from([
                        n.x + (i as f32) - 1.,
                        n.y + (j as f32) - 1.,
                        n.z + (k as f32) - 1.,
                    ]);
                    let icurr_x = curr_x.as_ivec3();
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                    // Get Mutex
                    let chunk_index = Chunk::node_world_pos_to_chunk_pos(icurr_x); 
                    let n_index = Chunk::node_world_pos_to_index(icurr_x);
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock();

                    // Do work inside mutex
                    let mut w_v = chunk[n_index].v;
                    
                    // Get the edge_mask for free while locking the first time
                    if i == 0 && j == 0 && k == 0 {
                        edge_mask = chunk.edge_mask;
                        if chunk[n_index].p.max(1) % p.p as u32 == 0 {
                            let mut dsb = despawn_buf.0.lock().unwrap();
                            dsb.push(e);
                            chunk[n_index].p /= p.p as u32;
                        }
                    }

                    std::mem::drop(chunk); // Unlock Mutex

                    w_v *= weight;
                    let term = Mat3A::from_cols(w_v * curr_dx.x, w_v * curr_dx.y, w_v * curr_dx.z);
                    b += term;
                    p.v += w_v;
                }
            }
        }
        p.c = b.mul_scalar(4.);

        let end = (sim_max_pos - (Chunk::EDGE_BUFFER_SIZE)) as f32;

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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for i in 0..sim_max_pos / Chunk::CHUNK_SIZE {
        for j in 0..sim_max_pos / Chunk::CHUNK_SIZE { 
            for k in 0..sim_max_pos / Chunk::CHUNK_SIZE { 
                grid.new_chunk(&IVec3::new(i as i32, j as i32, k as i32));
            }
        }
    }

    let blue_material = materials.add(StandardMaterial{
        base_color: BLUE_500.into(),
        unlit: true,
        ..Default::default()
    });
    let sphere_mesh = meshes.add(Sphere::new(0.2).mesh().ico(1).unwrap());


   // for x in 15..50 {
   //     for y in 15..50{
   //         for z in 15..50 {
   //             let pos = Vec3::new(x as f32, y as f32, z as f32);
   //             commands.spawn((
   //                 Particle {
   //                     x: pos,
   //                     v: Vec3::ZERO,
   //                     c: Mat3::ZERO,
   //                     p: 2,
   //                     m: 1.
   //                 },
   //                 Mesh3d(sphere_mesh.clone()),
   //                 MeshMaterial3d(blue_material.clone()),
   //                 Transform{translation: pos, ..Default::default()},
   //             ));
   //         }
   //     }
   // }
}

pub fn spawn(
    mut commands: Commands,
    spawn_state: ResMut<SpawnState>,
    spawn_buf: Res<SpawnBuffer>,
    despawn_buf: Res<DespawnBuffer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
    // Meshes for '5' particles
    let sphere_mesh = meshes.add(Sphere::new(0.2).mesh().ico(0).unwrap());
    let green_material = materials.add(StandardMaterial{
        base_color: LIME_500.into(),
        unlit: true,
        ..Default::default()
    });
    // Spawn any particles due to be spawned
    let mut sb = spawn_buf.0.lock().unwrap();

    for p in sb.drain(..) {
        // Change to spawn_batch
        commands.spawn((
                p.clone(),
                Mesh3d(sphere_mesh.clone()),
                MeshMaterial3d(green_material.clone()),
                Transform{translation: Vec3::from(p.x.clone()) , ..Default::default()},

        ));
    }
    drop(sb);

    // Despawn any particles due to be despawned
    let mut dsb = despawn_buf.0.lock().unwrap();
    for e in dsb.drain(..) {
        commands.entity(e).despawn();
    }
    drop(dsb);

    if spawn_state.0 == false {
        return;
    }

    let blue_material = materials.add(StandardMaterial{
        base_color: BLUE_500.into(),
        unlit: true,
        ..Default::default()
    });
    let sphere_mesh = meshes.add(Sphere::new(0.2).mesh().ico(0).unwrap());


    for x in 50..55 {
        for y in 50..55 {
            let pos = Vec3A::new(x as f32, y as f32, 5.);
            commands.spawn((
                    Particle {
                        x: pos,
                        v: Vec3A::new(0., 0., 2.),
                        c: Mat3A::ZERO,
                        p: 2,
                        m: 1.
                    },
                    Mesh3d(sphere_mesh.clone()),
                    MeshMaterial3d(blue_material.clone()),
                    Transform{translation: Vec3::from(pos), ..Default::default()},
                    ));
        }
    }

   // let orange_material = materials.add(StandardMaterial{
   //     base_color: ORANGE_500.into(),
   //     unlit: true,
   //     ..Default::default()
   // });
   // // Spawn particles as a spout at the top
   // commands.spawn((
   //     Particle {
   //         x: Vec3::new(8., 12., 8.),
   //         v: Vec3::ZERO,
   //         c: Mat3::ZERO,
   //         p: 3, 
   //         m: 1.
   //     },
   //     Mesh3d(sphere_mesh.clone()),
   //     MeshMaterial3d(orange_material.clone()),
   //     Transform{translation: Vec3::new(8., 12., 8.), ..Default::default()},

   // ));
}

pub fn draw(
    mut particles: Query<(&mut Particle, &mut Transform)>,
    mut gizmos: Gizmos,
    draw_state: Res<DrawState>,
) {
    particles.iter_mut().for_each(|(p, mut t)| {
        t.translation = Vec3::from(p.x);
    });

    if draw_state.0 == false {
        return;
    }
    gizmos.cuboid( 
        Transform::from_translation(Vec3::from((sim_max_pos as f32 / 2., sim_max_pos as f32 / 2., sim_max_pos as f32 / 2.))).with_scale(Vec3::splat(sim_max_pos as f32)),
        GHOST_WHITE
        );
    for i in 0..(sim_max_pos / Chunk::CHUNK_SIZE) + 1 {
        for j in 0..(sim_max_pos / Chunk::CHUNK_SIZE) + 1 {
            for k in 0..(sim_max_pos / Chunk::CHUNK_SIZE) + 1 {
                gizmos.sphere(Vec3::from((i as f32, j as f32, k as f32)) * 4., 0.05, GHOST_WHITE);
            }}}

}
