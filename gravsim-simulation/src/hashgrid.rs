use smallvec::SmallVec;
use std::ops::{AddAssign, Neg};

pub type ParticleId = usize;
pub type GridHash = usize;
pub type Coordinate = f32;

pub struct HashGrid {
    size_x: usize,
    size_y: usize,
    cell_size: Coordinate,
}

impl HashGrid {
    pub fn new(size_x: usize, size_y: usize, cell_size: Coordinate) -> Self {
        Self {
            size_x,
            size_y,
            cell_size,
        }
    }

    #[inline]
    pub fn get_hash_coordinate(&self, x: Coordinate, y: Coordinate) -> GridHash {
        ((x / self.cell_size).max(0.0) as GridHash).min(self.size_x - 1) * self.size_y
            + ((y / self.cell_size).max(0.0) as GridHash).min(self.size_y - 1)
    }

    #[inline]
    pub fn get_hash(&self, x: usize, y: usize) -> GridHash {
        x * self.size_y + y
    }

    #[inline]
    pub fn pos_from(&self, hash: GridHash) -> (usize, usize) {
        (hash / self.size_y, hash % self.size_x)
    }

    pub fn populate<P>(
        &mut self,
        particles: &[(Coordinate, Coordinate, &P)],
    ) -> Vec<Option<GridCell>> {
        let mut grid: Vec<Option<GridCell>> = std::iter::repeat(None)
            .take(self.size_x * self.size_y)
            .collect();

        for (id, (x, y, _)) in particles.iter().enumerate() {
            let hash = self.get_hash_coordinate(*x, *y);
            if let Some(cell) = &mut grid[hash] {
                cell.particles.push(id);
            } else {
                let mut cell = GridCell {
                    particles: SmallVec::new(),
                };
                cell.particles.push(id);
                grid[hash] = Some(cell);
            }
        }

        grid
    }

    pub fn interact<'a, P, F: AddAssign + Neg<Output = F> + Copy>(
        &mut self,
        particles: &[(Coordinate, Coordinate, &P)],
        forces_buffer: &mut [F],
        grid: &[Option<GridCell>],
        interact: impl Fn(&(Coordinate, Coordinate, &P), &(Coordinate, Coordinate, &P)) -> F,
    ) {
        assert_eq!(particles.len(), forces_buffer.len());
        let mut max = 0;

        for (cell_hash, cell) in grid
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.as_ref().map(|x| (i, x)))
        {
            let (cell_x, cell_y) = self.pos_from(cell_hash);

            max = max.max(cell.particles.len());

            for neighbour_x in cell_x.saturating_sub(1)..(cell_x + 2).min(self.size_x) {
                for neighbour_y in cell_y.saturating_sub(1)..(cell_y + 2).min(self.size_y) {
                    let neighbour_hash = self.get_hash(neighbour_x, neighbour_y);
                    if let Some(neighbour) = &grid[neighbour_hash] {
                        for i in 0..cell.particles.len() {
                            for j in 0..neighbour.particles.len() {
                                let cell_particle = cell.particles[i];
                                let neighbour_particle = neighbour.particles[j];

                                if cell_particle != neighbour_particle {
                                    let force = interact(
                                        &particles[cell_particle],
                                        &particles[neighbour_particle],
                                    );
                                    forces_buffer[cell_particle] += force;
                                    forces_buffer[neighbour_particle] += -force;
                                }
                            }
                        }
                    }
                }
            }
        }
        //println!("{}", max);
    }

    pub fn size_x(&self) -> usize {
        self.size_x
    }

    pub fn size_y(&self) -> usize {
        self.size_y
    }

    pub fn cell_size(&self) -> Coordinate {
        self.cell_size
    }
}

#[derive(Clone)]
pub struct GridCell {
    particles: SmallVec<[ParticleId; 8]>,
}
