use std::ops::{Deref, DerefMut};
use parking_lot::Mutex;
use crate::simulation::Node;

use bevy::utils::hashbrown::HashMap;
use bevy::prelude::*;

// TODO:
// Next major change is going to be transforming from AoS to SoA for better cache efficieny
// This is to see if the cache hits are more of a bottleneck than locking the grid mutex a second time
// If Cache miss time > (mutex lock * 2) + {constant index calculation}
// Then this will be beneficial and my thesis is that it will be significantly better for
// update_grid, slightly better for clear_grid, and slightly better for p2g1 and p2g2 and g2p
// because I beleive mutexes are lowly contested enough where mutex lock * 2 is less than cache
// miss
//

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
pub struct Chunk { 
    pub g: [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE],
    pub edge_mask: u8,
}

impl Chunk {
    // NOTE: It's assumed Chunk_size < i32_max && Chunk_Size > Edge_Buffer_Size
    // Consider changing to u32 or u16
    pub const CHUNK_SIZE: usize = 4; // Number of nodes in a chunk
    pub const EDGE_BUFFER_SIZE: usize = 2; // How big of boundary will edge conditions be applied to
    //pub const DX: f32 = 1.; // The difference between each node orthogonally

    /// 
    /// Input:
    /// World Pos of Node
    ///
    /// Output:
    /// Chunk coords of chunk responsible for that Node
    ///        
    /// Test Written and Passed
    #[inline]
    pub fn node_world_pos_to_chunk_pos(nwp: IVec3) -> IVec3 {
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
    /// Test Written and Passed
    #[inline]
    pub fn node_world_pos_to_index(nwp: IVec3) -> usize {
        let ichunk = Chunk::CHUNK_SIZE as i32;
        let nlp = nwp.rem_euclid(IVec3::splat(ichunk)); // Node World Position => Node Local Position 
        // TODO: See if Morton Encoding is more efficient
        nlp.z as usize * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE + nlp.y as usize * Chunk::CHUNK_SIZE + nlp.x as usize
    }

    ///
    /// Input:
    /// The local position of a Node
    ///
    /// Output:
    /// The edge mask of the current edges of the chunk a node is on
    /// According to the Edge_Buffer
    ///
    /// Test Written and Passed
    pub fn node_local_pos_to_buffer_edge_mask(nlp: UVec3) -> u8 {
        let mut edge_mask: u8 = 0;

        if (nlp.y as usize) < Chunk::EDGE_BUFFER_SIZE { edge_mask |= 1; } // 1
        else if (nlp.y as usize) > Chunk::CHUNK_SIZE - (Chunk::EDGE_BUFFER_SIZE + 1) { edge_mask |= 1 << 5; } // 6
        if (nlp.z as usize) < Chunk::EDGE_BUFFER_SIZE { edge_mask |= 1 << 1; } // 2
        else if (nlp.z as usize) > Chunk::CHUNK_SIZE - (Chunk::EDGE_BUFFER_SIZE + 1) { edge_mask |= 1 << 4; } // 5
        if (nlp.x as usize) < Chunk::EDGE_BUFFER_SIZE { edge_mask |= 1 << 2; } // 3
        else if (nlp.x as usize) > Chunk::CHUNK_SIZE - (Chunk::EDGE_BUFFER_SIZE + 1) { edge_mask |= 1 << 3; } // 4

        edge_mask
    }

    ///
    /// Input:
    /// The local position of a Node
    ///
    /// Output:
    /// The edge mask of the current edges of the chunk a node is on
    /// According to the buffer size of 1
    ///
    pub fn node_local_pos_to_one_edge_mask(nlp: UVec3) -> u8{
        let mut edge_mask: u8 = 0;

        if (nlp.y as usize) < 1 { edge_mask |= 1; } // 1
        else if (nlp.y as usize) > Chunk::CHUNK_SIZE - (1 + 1) { edge_mask |= 1 << 5; } // 6
        if (nlp.z as usize) < 1 { edge_mask |= 1 << 1; } // 2
        else if (nlp.z as usize) > Chunk::CHUNK_SIZE - (1 + 1) { edge_mask |= 1 << 4; } // 5
        if (nlp.x as usize) < 1 { edge_mask |= 1 << 2; } // 3
        else if (nlp.x as usize) > Chunk::CHUNK_SIZE - (1 + 1) { edge_mask |= 1 << 3; } // 4

        edge_mask
    }
    ///
    /// Input:
    /// The index of a node
    ///
    /// Output:
    /// The local position of the node inside its chunk
    ///
    /// Test Written and Passed
    pub fn index_to_node_local_pos(i: usize) -> UVec3 {
        let c2 = Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE;
        let z = i / c2;
        let y = (i % c2) / Chunk::CHUNK_SIZE;
        let x = i % Chunk::CHUNK_SIZE;
        
        UVec3::new(x as u32, y as u32, z as u32)
    }

    ///
    /// Input:
    /// Index of center node
    ///
    /// Output:
    /// An array from (-1, -1, -1) -> (1, 1, 1) centered around the center node
    ///
    /// Gives an array of neighboring indicies 
    /// Doesnt convert between index space and geometric space allowing for efficient translation
    /// Ordered x then y then z
    /// Follows this formula but more efficiently
    /// z as usize * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE + y as usize * Chunk::CHUNK_SIZE + x as usize
    pub fn neighbor_indices(i: u16) -> [u16; 27] {
        assert!(Chunk::CHUNK_SIZE == 4);

        // +-x = (i +- 1) % 4 + ((i >> 2) << 2)    
        //let plus_x = (i + 1) % 4 + ((i / 4) * 4);
        //let minus_x = (i - 1) % 4 + ((i / 4) * 4);
        let plus_x = (i + 1) & 3 | (i & !3);
        let minus_x = (i + 3) & 3 | (i & !3);

        // +-y = (i +- 4) % 16 + ((i >> 4) << 4)
        // let plus_y = (i + 4) % 16 + ((i / 16) * 16);
        // let minus_y = (i - 4) % 16 + ((i / 16) * 16);
        let plus_y = (i + 4) & 15 | (i & !15);
        let minus_y = (i + 12) % 16 | (i & !15);

        // +-z = (i +- 16) % 64 + ((i >> 6) << 6) 
        //let plus_z = (i + 16) % 64;
        //let minus_z = (i - 16) % 64;

        let plusx_plusy = (plus_y + 1) & 3 | (plus_y & !3);
        let plusx_miny = (minus_y + 1) & 3 | (minus_y & !3);
        let minx_plusy = (plus_y + 3) & 3 | (plus_y & !3);
        let minx_miny = (minus_y + 3) & 3 | (minus_y & !3);

        [
            ((minx_miny + 48) & 63),  (minus_y + 48) & 63,    (plusx_miny + 48) & 63,
            (minus_x + 48) & 63,    (i + 48) & 63,          (plus_x + 48) & 63,
            (minx_plusy + 48) & 63, (plus_y + 48) & 63,     (plusx_plusy + 48) & 63,

            minx_miny,              minus_y,                plusx_miny,
            minus_x,                i,                      plus_x,
            minx_plusy,             plus_y,                 plusx_plusy,

            (minx_miny + 16) & 63,  (minus_y + 16) & 63,    (plusx_miny + 16) & 63,
            (minus_x + 16) & 63,    (i + 16) & 63,          (plus_x + 16) & 63,
            (minx_plusy + 16) & 63, (plus_y + 16) & 63,     (plusx_plusy + 16) & 63
        ]
    }


    ///
    /// Input:
    /// The center node's edge mask
    /// The chunk the node is in
    ///
    /// Output:
    /// A list of IVec3 chunk positions of neighboring nodes
    ///
    /// Follows the standard of iteration established in this project`
    ///
    pub fn neighbor_chunks(chunk_pos: IVec3, node_index: u8) -> [IVec3; 27] {
        let node_local_positition = Chunk::index_to_node_local_pos(node_index as usize);
        let e = Chunk::node_local_pos_to_one_edge_mask(node_local_positition);
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
        let neg_y = IVec3::new(0, -(((e) & 1) as i32), 0 );
        let neg_z = IVec3::new(0, 0, -(((e >> 1) & 1) as i32));
        let neg_x = IVec3::new(-(((e >> 2) & 1) as i32), 0, 0);
        let pos_x = IVec3::new(((e >> 3) & 1) as i32, 0, 0);
        let pos_z = IVec3::new(0, 0, ((e >> 4) & 1) as i32);
        let pos_y = IVec3::new(0, ((e >> 5) & 1) as i32, 0);

        // TODO: See if pre-fetching mutex from grid here is faster
        [
            chunk_pos + neg_x + neg_y + neg_z,  chunk_pos + neg_y + neg_z,      chunk_pos + pos_x + neg_y + neg_z,
            chunk_pos + neg_x + neg_z,          chunk_pos + neg_z,              chunk_pos + pos_x + neg_z,               
            chunk_pos + neg_x + pos_y + neg_z,  chunk_pos + pos_y + neg_z,      chunk_pos + pos_x + pos_y + neg_z,

            chunk_pos + neg_x + neg_y,          chunk_pos + neg_y,              chunk_pos + pos_x + neg_y,
            chunk_pos + neg_x,                  chunk_pos,                      chunk_pos + pos_x,
            chunk_pos + neg_x + pos_y,          chunk_pos + pos_y,              chunk_pos + pos_x + pos_y,

            chunk_pos + neg_x + neg_y + pos_z,  chunk_pos + neg_y + pos_z,      chunk_pos + pos_x + neg_y + pos_z,
            chunk_pos + neg_x + pos_z,          chunk_pos + pos_z,              chunk_pos + pos_x + pos_z,
            chunk_pos + neg_x + pos_y + pos_z,  chunk_pos + pos_y + pos_z,      chunk_pos + pos_x + pos_y + pos_z,

        ]
    }

    pub fn new(edge_mask: u8) -> Self {
        Chunk{
            g: [Node::default(); Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE],
            edge_mask
        }
    }
}

impl Deref for Chunk {
    type Target = [Node; Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE * Chunk::CHUNK_SIZE];
    fn deref(&self) -> &Self::Target { &self.g } 
}

impl DerefMut for Chunk { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.g }  }

// Eulerian Grid Container 
#[derive(Resource)]
pub struct Grid (pub bevy::utils::HashMap<IVec3, Mutex<Chunk>>);

impl Grid {
    /// Inserts a new chunk at pos with its correct edge_mask
    /// Test Written and Passed
    pub fn new_chunk(&mut self, pos: &IVec3) {
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

        // If there is a chunk on the border update its edge mask
        // Otherwise update your own edge mask

        if let Some(c) = self.get(&(pos + IVec3::new(0, -1, 0))) {
            let mut chunk = c.lock();
            chunk.edge_mask &= 0b011111;
        } else {edge_mask |= 1;}
        if let Some(c) = self.get(&(pos + IVec3::new(0, 0, -1))) {
            let mut chunk = c.lock();
            chunk.edge_mask &= 0b101111;
        } else {edge_mask |= 1 << 1}
        if let Some(c) = self.get(&(pos + IVec3::new(-1, 0, 0))) {
            let mut chunk = c.lock();
            chunk.edge_mask &= 0b110111;
        } else {edge_mask |= 1 << 2}
        if let Some(c) = self.get(&(pos + IVec3::new(1, 0, 0))) {
            let mut chunk = c.lock();
            chunk.edge_mask &= 0b111011;
        } else {edge_mask |= 1 << 3}
        if let Some(c) = self.get(&(pos + IVec3::new(0, 0, 1))) {
            let mut chunk = c.lock();
            chunk.edge_mask &= 0b111101;
        } else {edge_mask |= 1 << 4}
        if let Some(c) = self.get(&(pos + IVec3::new(0, 1, 0))) {
            let mut chunk = c.lock();
            chunk.edge_mask &= 0b111110;
        } else {edge_mask |= 1 << 5}
        
        let chunk = Chunk::new(edge_mask);
        self.insert(*pos, Mutex::new(chunk));
    }
}

impl Deref for Grid {
    type Target = HashMap<IVec3, Mutex<Chunk>>;
    fn deref(&self) -> &Self::Target { &self.0 } 
}
impl DerefMut for Grid { fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn node_world_pos_to_chunk_pos_test() {
        let n = Chunk::CHUNK_SIZE as i32;

        // 0, 0, 0 should be responsible for 0, 0, 0 up to (n - 1, n - 1, n - 1)
        assert!(Chunk::node_world_pos_to_chunk_pos(IVec3::ZERO) == IVec3::ZERO);
        assert!(Chunk::node_world_pos_to_chunk_pos(IVec3::splat(n - 1)) == IVec3::ZERO);

        // -1, -1, -1 should be responsible for (-n, -n, -n) up to (-1, -1, -1)
        assert!(Chunk::node_world_pos_to_chunk_pos(IVec3::splat(-1)) == IVec3::splat(-1));
        assert!(Chunk::node_world_pos_to_chunk_pos(IVec3::splat(-n)) == IVec3::splat(-1));

        // 1, 1, 1 should be responsible for (n, n, n) up to (2n - 1, 2n - 1, 2n -1)
        assert!(Chunk::node_world_pos_to_chunk_pos(IVec3::splat(n)) == IVec3::splat(1));
        assert!(Chunk::node_world_pos_to_chunk_pos(IVec3::splat((2 * n) - 1)) == IVec3::splat(1));
    }

    #[test]
    pub fn node_world_pos_to_index_test_linear() {
        let n = Chunk::CHUNK_SIZE as i32;

        // (0, 0, 0) should be 0
        assert!(Chunk::node_world_pos_to_index(IVec3::ZERO) == 0);
        // (n, n, n) should be 0
        assert!(Chunk::node_world_pos_to_index(IVec3::splat(n)) == 0);
        // (0, 0, 1) should be 1
        assert!(Chunk::node_world_pos_to_index(IVec3::from((1, 0, 0))) == 1);
        // (-n, -n, -n) should be 0
        assert!(Chunk::node_world_pos_to_index(IVec3::splat(-n)) == 0);
        // (-1, -1, -1) should be n * n * n
        assert!(Chunk::node_world_pos_to_index(IVec3::splat(-1)) == (n * n * n - 1) as usize);
    }

    #[test]
    fn node_local_pos_to_buffer_edge_mask_test() {
        let n = Chunk::CHUNK_SIZE as u32;

        // (0, 0, 0) should be 111
        assert!(Chunk::node_local_pos_to_buffer_edge_mask(UVec3::ZERO) == 0b111);
        // (n -1, n - 1, n - 1) should be 111000
        assert!(Chunk::node_local_pos_to_buffer_edge_mask(UVec3::splat(n - 1)) == 0b111000);
        // (n - 1, 0, 0) should be 1011
        assert!(Chunk::node_local_pos_to_buffer_edge_mask(UVec3::new(n - 1, 0, 0)) == 0b1011);
    }

    #[test]
    fn index_to_node_local_pos() {
        let n = Chunk::CHUNK_SIZE as u32;

        // 0 should be (0, 0, 0)
        assert!(Chunk::index_to_node_local_pos(0) == UVec3::ZERO);
        // n * n * n - 1 should be n - 1, n - 1, n - 1
        assert!(Chunk::index_to_node_local_pos((n * n * n) as usize - 1) == UVec3::splat(n - 1));
        // n - 1 should be (n - 1, 0, 0)
        assert!(Chunk::index_to_node_local_pos((n) as usize - 1) == UVec3::new(n - 1, 0, 0));
    }

    #[test]
    fn single_grid_chunk() {
        let mut grid = Grid(HashMap::new());
        grid.new_chunk(&IVec3::ZERO);

        let chunk = grid.get(&IVec3::ZERO).unwrap().lock();
        assert!(chunk.edge_mask == 0b111111)
    }

    #[test]
    fn orthogonal_grid_chunk() {
        let mut grid = Grid(HashMap::new());
        grid.new_chunk(&IVec3::ZERO);
        grid.new_chunk(&IVec3::new(0, 0, 1));
        grid.new_chunk(&IVec3::new(0, 1, 0));
        grid.new_chunk(&IVec3::new(1, 0, 0));
        grid.new_chunk(&IVec3::new(0, 0, -1));
        grid.new_chunk(&IVec3::new(0, -1, 0));
        grid.new_chunk(&IVec3::new(-1, 0, 0));

        let chunk0 = grid.get(&IVec3::ZERO).unwrap().lock();
        let chunk1 = grid.get(&IVec3::new(0, 0, 1)).unwrap().lock();
        let chunk2 = grid.get(&IVec3::new(0, 1, 0)).unwrap().lock();
        let chunk3 = grid.get(&IVec3::new(1, 0, 0)).unwrap().lock();
        let chunk4 = grid.get(&IVec3::new(0, 0, -1)).unwrap().lock();
        let chunk5 = grid.get(&IVec3::new(0, -1, 0)).unwrap().lock();
        let chunk6 = grid.get(&IVec3::new(-1, 0, 0)).unwrap().lock();
        println!("{:b}", chunk0.edge_mask);
        assert!(chunk0.edge_mask == 0b0);
        assert!(chunk1.edge_mask == 0b111101);
        assert!(chunk2.edge_mask == 0b111110);
        assert!(chunk3.edge_mask == 0b111011);
        assert!(chunk4.edge_mask == 0b101111);
        assert!(chunk5.edge_mask == 0b11111);
        assert!(chunk6.edge_mask == 0b110111);
    }

    #[test]
    fn node_neighbor_test() {
        for i in 0..64 {
            let neighbors = Chunk::neighbor_indices(i);

            let curr_node = Chunk::index_to_node_local_pos(i as usize).as_ivec3();

            let mut index = 0;
            for k in 0..3 {
                for j in 0..3 {
                    for i in 0..3 {
                        let curr_x = curr_node + IVec3::new(i, j, k) - IVec3::ONE;
                        assert!( Chunk::node_world_pos_to_index(curr_x) == neighbors[index as usize] as usize );
                        index += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_neighbor_test() {
        for i in 0..64 {
            let current_node = Chunk::index_to_node_local_pos(i).as_ivec3();
            let chunk_pos = Chunk::node_world_pos_to_chunk_pos(current_node); 

            let neighbors = Chunk::neighbor_chunks(chunk_pos, i as u8);

            let curr_node = Chunk::index_to_node_local_pos(i as usize).as_ivec3();
            let mut index = 0;
            for k in 0..3 {
                for j in 0..3 {
                    for i in 0..3 {
                        let curr_x = curr_node + IVec3::new(i, j, k) - IVec3::ONE;
                        assert!( Chunk::node_world_pos_to_chunk_pos(curr_x) == neighbors[index as usize] );
                        index += 1;
                    }
                }
            }
        }
    }
}
