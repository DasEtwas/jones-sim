
use std::ops::{ AddAssign, Neg};
use bumpalo::Bump;

pub type ParticleId = usize;
pub type GridHash = usize;

pub struct HashGrid<'a> {
    grid: Vec<Option<GridCell<'a>>>,
    grid_alloc: Bump,
    size_x: usize,
    size_y: usize,
    cell_size: f64,
}

impl<'a> HashGrid<'a> {
    pub fn new(size_x: usize, size_y: usize, cell_size: f64) -> Self {
        Self {
            grid_alloc: Bump::with_capacity(size_x * size_y * 2),
            grid: std::iter::repeat(None).take(size_x * size_y).collect(),
            size_x,
            size_y,
            cell_size,
        }
    }

    #[inline]
    pub fn get_hash_f64(&self, x: f64, y: f64) -> GridHash {
        (x / self.cell_size) as GridHash * self.size_y + (y / self.cell_size) as GridHash
    }

    #[inline]
    pub fn get_hash(&self, x: usize, y: usize) -> GridHash {
        x * self.size_y + y
    }

    #[inline]
    pub fn pos_from(&self, hash: GridHash) -> (usize, usize) {
        (hash / self.size_y, hash % self.size_x)
    }

    pub fn populate<P, F: AddAssign + Neg<Output=F> + Copy>(&'a mut self, particles: &[(f64, f64, &P)], forces_buffer: &mut [F], interact: impl Fn(&(f64,f64,&P), &(f64,f64,&P)) -> F) {
        assert_eq!(particles.len(), forces_buffer.len());

        for (id, (x, y, _)) in particles.iter().enumerate() {
            let hash = self.get_hash_f64(*x, *y);
            if let Some(cell) = &mut self.grid[hash] {
                cell.particles.push(id);
            } else {
                let mut cell = GridCell { particles: bumpalo::collections::Vec::new_in(&self.grid_alloc) };
                cell.particles.push(id);
                self.grid[hash] = Some(cell);
            }
        }

        for (cell_hash, cell) in self.grid.iter().enumerate().filter_map(|(i, x)| x.as_ref().map(|x| (i, x))) {
            let (cell_x, cell_y) = self.pos_from(cell_hash);

            for neighbour_x in cell_x.saturating_sub(1)..=(cell_x + 1).min(self.size_x) {
                for neighbour_y in cell_y.saturating_sub(1)..=(cell_y + 1).min(self.size_y) {
                    let neighbour_hash = self.get_hash(neighbour_x, neighbour_y);
                    if let Some(neighbour) = &self.grid[neighbour_hash] {
                        for i in 0..cell.particles.len() {
                            let start = i + 1;

                            for j in start..neighbour.particles.len() {
                                let cell_particle = cell.particles[i];
                                let neighbour_particle = neighbour.particles[j];

                                if cell_particle != neighbour_particle {
                                    let force = interact(&particles[cell_particle], &particles[neighbour_particle]);
                                    forces_buffer[cell_particle] += force;
                                    forces_buffer[neighbour_particle] += -force;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.grid.iter_mut().for_each(|c| *c = None);
    }
}

#[derive(Clone)]
pub struct GridCell<'alloc> {
    particles: bumpalo::collections::Vec<'alloc, ParticleId>,
}