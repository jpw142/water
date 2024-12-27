use rayon::prelude::*;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::collections::HashMap;
use bevy::prelude::*;

const dt: f32 = 0.4;
const gravity: f32 = -0.3;
const rest_density: f32 = 4.0;
const dynamic_viscosity: f32 = 0.1;
const eos_stiffness: f32 = 10.0;
const eos_power: f32 = 4.;


/// Langrangian Particle
#[derive(Component)]
struct Particle {
    x: Vec3,    // Position
    v: Vec3,    // Velocity
    c: Mat3,    // Affine Momentum Matrix
    m: f32,     // Mass
    p: u16    // material prime index
}

/// Eulerian Grid Node
#[derive(Copy, Clone, Default)]
struct Node {
    v: Vec3,    // Velocity
    m: f32,     // Mass
    m_old: f32,     // Mass
    p: u32,     // Material Number
}

impl Node {
    fn zero(&mut self) {
        self.p = 0;
        self.m_old = self.m;
        self.m = 0.0;
        self.v = Vec3::ZERO;
    }
}

///
/// A chunk is an N x N x N block of eulerian nodes that make up a larger fluid simulation space
/// N = Chunk_Size
///
/// A node is uniquely identified in a chunk by its local coordinates (0..N, 0..N, 0..N)
/// These coordinates get flattened down into a 1d usize index and accessed in the array
///
/// A chunk is uniquely identified in a Grid by it's chunk coordinates (x, y, z) 
/// These coordinate are accessed through a hashmap
///
/// The chunk is responsible for Nodes from (x*N, y*N, z*N) to (((x+1)*N)-1, ((y+1)*N)-1, ((z+1)*N)-1)
/// EXAMPLE for N = 16:
/// Chunk (0, 0, 0) is responsible for (0, 0, 0) -> (15, 15, 15)
/// Chunk (-1, -1, -1) is responsible for (-16, -16, -16) -> (-1, -1, -1)
///
struct Chunk { 
    g: [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE],
    edge_mask: u8,
}

impl Chunk {
    // NOTE: It's assumed Chunk_size < i32_max && Chunk_Size > Edge_Buffer_Size
    const CHUNK_SIZE: usize = 16; // Number of nodes in a chunk
    const EDGE_BUFFER_SIZE: usize = 2; // How big of boundary will edge conditions be applied to
    const DX: f32 = 1.; // The difference between each node orthogonally

    /// 
    /// Input:
    /// World Pos of Node
    ///
    /// Output:
    /// Chunk coords of chunk responsible for that Node
    //        
    fn node_world_pos_to_chunk_pos(nwp: IVec3) -> IVec3 {
        let ichunk = Chunk::CHUNK_SIZE as i32; 
        // Adjusts for off by 1 for negatives because np 0,0,0 is in cp 0,0,0
        IVec3::new(
            if nwp.x < 0 { (nwp.x + 1) / ichunk - 1 } else { nwp.x / ichunk },
            if nwp.y < 0 { (nwp.y + 1) / ichunk - 1 } else { nwp.y / ichunk },
            if nwp.z < 0 { (nwp.z + 1) / ichunk - 1 } else { nwp.z / ichunk }
        )
    }

    ///
    /// Input:
    /// World Pos of Node
    ///
    /// Output:
    /// Index of node in its chunk
    ///
    fn node_world_pos_to_index(nwp: IVec3) -> usize {
        let ichunk = Chunk::CHUNK_SIZE as i32;
        let nlp = nwp.rem_euclid(ichunk); // Node World Position => Node Local Position 
        // TODO: See if Morton Encoding is more efficient
        nlp.z as usize * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE + nlp.y as usize * Chunk::CHUNK_SIZE + nlp.x as usize
    }

    ///
    /// Input:
    /// The local position of a Node
    ///
    /// Output:
    /// The edge mask of the current edges of the chunk a node is on
    ///
    fn node_local_pos_to_edge_mask(nlp: UVec3) -> u8 {
        let mut edge_mask: u8 = 0;

        // +1 for off by 1 by 0 index and +1 due to always rounding particle position down
        if y < Chunk::EDGE_BUFFER_SIZE { edge_mask | 1 } // 1
        else if y > Chunk::CHUNK_SIZE - (Chunk::EDGE_BUFFER_SIZE + 2) { edge_mask | 1 << 5 } // 6
        if z < Chunk::EDGE_BUFFER_SIZE { edge_mask | 1 << 1 } // 2
        else if z > Chunk::CHUNK_SIZE - (Chunk::EDGE_BUFFER_SIZE + 2) { edge_mask | 1 << 4 } // 5
        if x < Chunk::EDGE_BUFFER_SIZE { edge_mask | 1 << 2 } // 3
        else if x > Chunk::CHUNK_SIZE - (Chunk::EDGE_BUFFER_SIZE + 2) { edge_mask | 1 << 3 } // 4

        edge_mask
    }

    fn index_to_node_local_pos(i: usize) -> UVec3 {
        let c2 = Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE;
        let z = i / c2;
        let y = (i % c2) / Chunk::CHUNK_SIZE;
        let x = i % Chunk::CHUNK_SIZE;
        
        UVec3::new(x, y, z)
    }
}
impl Deref for Chunk {
    type Target = [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE];
    fn deref(&self) -> &Self::Target { &self.g } 
}
impl DerefMut for Chunk { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.g }  }

// Eulerian Grid Container 
#[derive(Resource)]
struct Grid (HashMap<(i32, i32, i32), Mutex<Chunk>>);
impl Deref for Grid {
    type Target = HashMap<(i32, i32, i32), Mutex<Chunk>>;
    fn deref(&self) -> &Self::Target { &self.0 } 
}
impl DerefMut for Grid { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 } }

impl Grid {
    fn new_chunk(&mut self, pos: &(i32, i32, i32)) {
        // If the chunk is already in the hashmap
        if let Some(_) = self.get(pos) {
            panic!("used new_chunk for already initilized chunk");
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
        let mut edge_mask = 0;
        // 1
        if let None = self.get(&(pos.0 , pos.1 - 1, pos.2)) {
            edge_mask |= 1;
        }
        // 2
        if let None = self.get(&(pos.0 , pos.1, pos.2 - 1)) {
            edge_mask |= 1 << 1;
        }
        // 3 
        if let None = self.get(&(pos.0 - 1, pos.1, pos.2)) {
            edge_mask |= 1 << 2;
        }
        // 4
        if let None = self.get(&(pos.0 + 1, pos.1, pos.2)) {
            edge_mask |= 1 << 3;
        }
        // 5
        if let None = self.get(&(pos.0, pos.1, pos.2 + 1)) {
            edge_mask |= 1 << 4;
        }
        // 6
        if let None = self.get(&(pos.0, pos.1 + 1, pos.2)) {
            edge_mask |= 1 << 5;
        }
        let chunk = Chunk{
            g: [Node::default(); Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE],
            edge_mask
        };

        self.insert(*pos, Mutex::new(chunk));

    }
}

/// Sets all nodes in the grid to 0
fn clear_grid ( mut grid: ResMut<Grid> ) {
    grid.par_iter_mut().for_each(|(_, c)| {
        let mut chunk = c.lock().expect("Error locking Mutex for clear_grid");
        chunk.iter_mut().for_each(|node| {
            node.zero();
        });
    });
}

// First Transfer of Langrangian Particle Information to Eulerian Grid
fn p2g (
    grid: Res<Grid>,
    particles: Query<&Particle>
) {
    particles.par_iter().for_each(|p| {
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
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n

                    
                    // Pre-calculate values we will send to mutex
                    let q = p.c * curr_dx;
                    let mass_contrib = weight * p.m;
                    let v_contrib = mass_contrib * (p.v + Vec3::from(q));

                    // Get Mutex
                    let chunk_index = Chunk::pos_to_chunk_index(curr_x); 
                    let n_index = Chunk::pos_to_index(curr_x);
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock().expect("Error locking mutex for p2g");

                    // Do work inside mutex
                    chunk[n_index].m += mass_contrib;
                    chunk[n_index].v += v_contrib;
                    density += chunk[n_index].m_old * weight;

                    std::mem::drop(chunk); // Unlock Mutex
                }
            }
        }
        let volume = p.m / (density + 0.00000001);
        let pressure = (-0.1_f32).max(eos_stiffness * (density / rest_density).powf(eos_power) - 1.);
        let mut stress = Mat3::from_cols_array(&[
                                               -pressure, 0., 0., 
                                               0., -pressure, 0.,
                                               0., 0., -pressure,
        ]);
        let mut strain = p.c;

        let trace = strain.x_axis.z + strain.y_axis.y + strain.z_axis.x;
        strain.x_axis.z = trace;
        strain.y_axis.y = trace;
        strain.z_axis.x = trace;

        let viscosity_term: Mat3 = dynamic_viscosity * strain;
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
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                    let momentum = Vec3::from(eq_16_term_0 * weight * curr_dx);
                    // Get Mutex
                    let chunk_index = Chunk::pos_to_chunk_index(curr_x); 
                    let n_index = Chunk::pos_to_index(curr_x);
                    let mut chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock().expect("Error locking mutex for p2g");

                    // Do work inside mutex
                    chunk[n_index].v += momentum;

                    std::mem::drop(chunk); // Unlock Mutex
                }
            }
        }
    })
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
fn update_grid ( grid: Res<Grid> ) {
    grid.par_iter().for_each(|(_, c)| {
        let mut chunk = c.lock().expect("Error locking Mutex for update_grid");
        let edge_mask = chunk.edge_mask;
        let end1 = Chunk::CHUNK_SIZE - 1;
        let end2 = Chunk::CHUNK_SIZE - 2;
        chunk.iter_mut().enumerate().for_each(|(i, node)| {
            // Boundary condition
            let (x, y, z) = Chunk::index_to_pos(i);
            if (edge_mask & 1) != 0 { if y == 0 || y == 1 {node.v = Vec3::ZERO} } // 1
            if (edge_mask & 1 << 1) != 0 { if z == 0 || z == 1 {node.v = Vec3::ZERO} } // 2
            if (edge_mask & 1 << 2) != 0 { if x == 0 || x == 1 {node.v = Vec3::ZERO} } // 3
            if (edge_mask & 1 << 3) != 0 { if x == end1 || x == end2 {node.v = Vec3::ZERO} } //4 
            if (edge_mask & 1 << 4) != 0 { if z == end1 || z == end2 {node.v = Vec3::ZERO} } // 5
            if (edge_mask & 1 << 5) != 0 { if y == end1 || y == end2 {node.v = Vec3::ZERO} } // 6

            if node.m > 0. && node.v != Vec3::ZERO {
                node.v /= node.m;
                node.v.y += dt * gravity;
            }
        });
    });
}

fn g2p (
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
                    let curr_dx = (curr_x - p.x) + 0.5; // p distance to current n
                    // Get Mutex
                    let chunk_index = Chunk::pos_to_chunk_index(curr_x); 
                    let n_index = Chunk::pos_to_index(curr_x);
                    let chunk = grid.get(&chunk_index).expect("Particle somehow out of bounds of loaded chunks").lock().expect("Error locking mutex for p2g");

                    // Do work inside mutex
                    let mut w_v = chunk[n_index].v;
                    if i == 1 && j == 1 && k == 1 {
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

        let end = (Chunk::CHUNK_SIZE - 2) as f32;

        if (edge_mask & 1) != 0 { if n.y < 2. {p.v.y *= -1.} } // 1
        if (edge_mask & 1 << 1) != 0 { if n.z < 2. {p.v.z *= -1.} } // 2
        if (edge_mask & 1 << 2) != 0 { if n.x < 2. {p.v.x *= -1.} } // 3
        if (edge_mask & 1 << 3) != 0 { if n.x > end {p.v.x *= -1.} } //4 
        if (edge_mask & 1 << 4) != 0 { if n.z > end {p.v.z *= -1.} } // 5
        if (edge_mask & 1 << 5) != 0 { if n.y > end {p.v.y *= -1.} } // 6

        let v = p.v;
        p.x += v * dt;
    })
}

fn initialize(
    mut commands: Commands,
    mut grid: ResMut<Grid>,
) {
    grid.new_chunk(&(0,0,0));
    for i in 0..10 {
        commands.spawn(Particle{
            x: Vec3::new(4., 4., 4.),
            v: Vec3::ZERO,
            c: Mat3::ZERO,
            p: 0,
            m: 1.
        });
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(Grid(HashMap::new()))
        .add_systems(Startup, initialize)
        .add_systems(Update, (clear_grid, p2g, update_grid, g2p).chain())
        .run();
    println!("{}", -0);
    println!("{}", size_of::<Node>());
}
