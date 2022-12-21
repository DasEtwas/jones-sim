use smallvec::SmallVec;
use std::ops::{AddAssign, Neg};
use std::sync::atomic::{AtomicU16, Ordering};

pub type ParticleId = usize;
pub type GridHash = usize;
pub type Coordinate = f32;

pub struct HashGrid {
    size_x: usize,
    size_y: usize,
    cell_size: Coordinate,
    periodic: bool,
}

impl HashGrid {
    pub fn new(size_x: usize, size_y: usize, cell_size: Coordinate, periodic: bool) -> Self {
        Self {
            size_x,
            size_y,
            cell_size,
            periodic,
        }
    }

    #[inline]
    pub fn get_hash_coordinate(&self, x: Coordinate, y: Coordinate) -> GridHash {
        self.get_hash(
            ((x / self.cell_size).max(0.0) as GridHash).min(self.size_x - 1),
            ((y / self.cell_size).max(0.0) as GridHash).min(self.size_y - 1),
        )
    }

    #[inline]
    pub fn get_hash(&self, x: usize, y: usize) -> GridHash {
        x * self.size_y + y
    }

    #[inline]
    pub fn pos_from(&self, hash: GridHash) -> (usize, usize) {
        (hash / self.size_y, hash % self.size_y)
    }

    pub fn populate<P>(
        &mut self,
        particles: &[(Coordinate, Coordinate, &P)],
    ) -> Vec<Option<GridCell>> {
        let mut grid: Vec<Option<GridCell>> = std::iter::from_fn(|| Some(None))
            .take(self.size_x * self.size_y)
            .collect();

        for (id, (x, y, _)) in particles.iter().enumerate() {
            let hash = self.get_hash_coordinate(*x, *y);
            if let Some(cell) = &mut grid[hash] {
                cell.particles.push(id);
            } else {
                let mut cell = GridCell {
                    particles: SmallVec::new(),
                    interacted_with: AtomicU16::new(0),
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
        interact: impl Fn(Coordinate, Coordinate, &P, &P) -> F,
    ) {
        assert_eq!(particles.len(), forces_buffer.len());
        let mut max = 0;

        let size_x = self.size_x;
        let size_y = self.size_y;
        let cell_size = self.cell_size;

        let size_x_f = size_x as f32 * cell_size;
        let size_y_f = size_y as f32 * cell_size;

        let periodic = self.periodic;

        let non_empty_cells = grid
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.as_ref().map(|x| (i, x)))
            .collect::<Vec<_>>();

        for (cell_hash, cell) in non_empty_cells {
            let (cell_x, cell_y) = self.pos_from(cell_hash);

            max = max.max(cell.particles.len());

            if periodic {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let neighbour_hash = self.get_hash(
                            (cell_x as i32 + dx).rem_euclid(size_x as i32) as usize,
                            (cell_y as i32 + dy).rem_euclid(size_y as i32) as usize,
                        );
                        if !cell.has_interacted_with(dx, dy) {
                            if let Some(neighbour) = &grid[neighbour_hash] {
                                if !neighbour.has_interacted_with(-dx, -dy) {
                                    cell.mark_interacted(dx, dy);
                                    neighbour.mark_interacted(-dx, -dy);

                                    for cell_particle in &cell.particles {
                                        let a = &particles[*cell_particle];

                                        for neighbour_particle in &neighbour.particles {
                                            if cell_particle != neighbour_particle {
                                                let b = &particles[*neighbour_particle];

                                                let dx = if cell_x == 0 && dx == -1 {
                                                    b.0 - a.0 - size_x_f
                                                } else if cell_x + 1 == self.size_x && dx == 1 {
                                                    b.0 - a.0 + size_x_f
                                                } else {
                                                    b.0 - a.0
                                                };

                                                let dy = if cell_y == 0 && dy == -1 {
                                                    b.1 - a.1 - size_y_f
                                                } else if cell_y + 1 == self.size_y && dy == 1 {
                                                    b.1 - a.1 + size_y_f
                                                } else {
                                                    b.1 - a.1
                                                };

                                                let force = interact(dx, dy, a.2, b.2);
                                                forces_buffer[*cell_particle] += force;
                                                forces_buffer[*neighbour_particle] += -force;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for neighbour_x in cell_x.saturating_sub(1)..(cell_x + 2).min(size_x) {
                    for neighbour_y in cell_y.saturating_sub(1)..(cell_y + 2).min(size_y) {
                        let neighbour_hash = self.get_hash(neighbour_x, neighbour_y);
                        if !cell.has_interacted_with(
                            neighbour_x as i32 - cell_x as i32,
                            neighbour_y as i32 - cell_y as i32,
                        ) {
                            if let Some(neighbour) = &grid[neighbour_hash] {
                                if !neighbour.has_interacted_with(
                                    cell_x as i32 - neighbour_x as i32,
                                    cell_y as i32 - neighbour_y as i32,
                                ) {
                                    cell.mark_interacted(
                                        neighbour_x as i32 - cell_x as i32,
                                        neighbour_y as i32 - cell_y as i32,
                                    );
                                    neighbour.mark_interacted(
                                        cell_x as i32 - neighbour_x as i32,
                                        cell_y as i32 - neighbour_y as i32,
                                    );

                                    for cell_particle in &cell.particles {
                                        let a = &particles[*cell_particle];

                                        for neighbour_particle in &neighbour.particles {
                                            if cell_particle != neighbour_particle {
                                                let b = &particles[*neighbour_particle];

                                                let force =
                                                    interact(b.0 - a.0, b.1 - a.1, a.2, b.2);
                                                forces_buffer[*cell_particle] += force;
                                                forces_buffer[*neighbour_particle] += -force;
                                            }
                                        }
                                    }
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

    pub fn periodic(&self) -> bool {
        self.periodic
    }
}

pub struct GridCell {
    particles: SmallVec<[ParticleId; 8]>,
    interacted_with: AtomicU16,
}

impl GridCell {
    pub fn has_interacted_with(&self, dx: i32, dy: i32) -> bool {
        (self.interacted_with.load(Ordering::Relaxed) & (1u16 << Self::bit_index(dx, dy))) != 0
    }

    pub fn mark_interacted(&self, dx: i32, dy: i32) {
        self.interacted_with
            .fetch_or(1u16 << Self::bit_index(dx, dy), Ordering::Relaxed);
    }

    fn bit_index(dx: i32, dy: i32) -> usize {
        (dx + 1) as usize + (dy + 1) as usize * 3
    }
}
