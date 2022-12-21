use crate::hashgrid::HashGrid;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

pub mod hashgrid;

#[derive(Copy, Clone, Debug)]
pub struct Atom {
    pub inv_mass: f32,
    pub radius: f32,
    pub pos: Vector2<f32>,
    pub vel: Vector2<f32>,
    pub force: Vector2<f32>,
    pub color: [f32; 3],
}

impl Atom {
    pub const DENSITY: f32 = 250.0;

    pub fn new(pos: Vector2<f32>, vel: Vector2<f32>, color: [f32; 3], mass: f32) -> Self {
        Self {
            pos,
            inv_mass: mass.recip(),
            radius: 1.0,
            force: Vector2::zeros(),
            vel,
            color,
        }
    }
}

/// Represents a mass point in space.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MassData {
    pub position: Vector2<f32>,
    pub mass: f32,
}

pub struct Simulation {
    pub stars: Vec<Atom>,
    pub forces_buf: Vec<Vector2<f32>>,
    pub grid: HashGrid,

    pub jones_lut: Vec<f32>,
    pub jones_lut_fac: f32,
}

/// Positive output means there is an attractive force
/// The output force is premultiplied with `1/sqrt(dist-sq)`
#[inline]
fn lennard_jones_normalizing(dist_sq: f32) -> f32 {
    const DESIRED_RADIUS: f32 = 1.0;
    const SIGMA_FAC: f32 = 1.122462048309373; // 6th root of 2, the factor of the root relative to sigma
    const SIGMA: f32 = DESIRED_RADIUS / SIGMA_FAC;
    const SIGMA6: f32 = SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA; // precomputed sigma^6
    const E: f32 = 0.25 * 3.0;

    (((24.0 * E * SIGMA6 * (dist_sq.powi(3) - 2.0 * SIGMA6)) / dist_sq.powi(7)) as f32).max(-1e7)
}

impl Simulation {
    pub fn new<I>(stars: I, side_length: f32, cell_size: f32, margin: f32, periodic: bool) -> Self
    where
        I: IntoIterator<Item = Atom>,
    {
        let stars: Vec<Atom> = stars.into_iter().collect();

        let lut_size = 256;
        let lut_max_function_value = 16.0;
        let lut_fac = lut_max_function_value / lut_size as f32;
        let mut lut = vec![];
        for i in 0..lut_size {
            let f = i as f32 * lut_fac;
            lut.push(lennard_jones_normalizing(f));
        }

        if periodic {
            assert_eq!(
                margin, 0.0,
                "A periodic field may not have a grid size margin"
            );
        }

        Self {
            forces_buf: vec![Vector2::<f32>::zeros(); stars.len()],
            stars,
            grid: HashGrid::new(
                (side_length * (1.0 + margin) / cell_size) as usize,
                (side_length * (1.0 + margin) / cell_size) as usize,
                cell_size,
                periodic,
            ),
            jones_lut_fac: lut_fac.recip(),
            jones_lut: lut,
        }
    }

    pub fn update(&mut self) {
        let particles = self
            .stars
            .iter()
            .map(|star| (star.pos.x, star.pos.y, &()))
            .collect::<Vec<_>>();

        #[inline]
        fn interact(dx: f32, dy: f32, _: &(), _: &()) -> Vector2<f32> {
            let dist_sq = dx * dx + dy * dy;

            let f = lennard_jones_normalizing(dist_sq);

            Vector2::new(f * dx, f * dy)
        }

        // slower lol
        let interact_lut = |dx: f32, dy: f32, _: &(), _: &()| -> Vector2<f32> {
            let dist_sq = dx * dx + dy * dy;

            let lut = *self
                .jones_lut
                .get((dist_sq * self.jones_lut_fac) as usize)
                .unwrap_or(&0.0);

            // we multiply by 0.5 because we touch every particle twice in interaction (boo!)
            let f = lut;

            Vector2::new(f * dx, f * dy)
        };

        let mut grid = self.grid.populate(&particles);
        self.grid
            .interact(&particles, &mut self.forces_buf, &mut grid, interact);

        //let damping = 0.0005;
        let damping = 0.0001;
        //let damping = 0.0001;

        self.stars
            .iter_mut()
            .zip(&mut self.forces_buf)
            .for_each(|(star, force)| {
                star.force = force.cast::<f32>();
                star.vel += star.force * star.inv_mass;
                star.vel *= 1.0 - damping;
                star.pos += 1e-6 * star.vel;

                if self.grid.periodic() {
                    star.pos.x = star
                        .pos
                        .x
                        .rem_euclid(self.grid.size_x() as f32 * self.grid.cell_size());
                    star.pos.y = star
                        .pos
                        .y
                        .rem_euclid(self.grid.size_y() as f32 * self.grid.cell_size());
                }

                *force = Vector2::zeros();
            });
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MassDistribution {
    alpha: f32,
    max_mass: f32,
}

impl MassDistribution {
    pub fn new(alpha: f32, max_mass: f32) -> Self {
        Self { alpha, max_mass }
    }
}

impl MassDistribution {
    pub fn sample(&self, t: f32) -> f32 {
        self.max_mass * ((self.alpha * t).exp_m1() / self.alpha.exp_m1()).min(1.0)
    }

    pub fn eval_inv(&self, x: f32) -> f32 {
        (self.alpha.exp_m1() * x / self.max_mass + 1.0).ln() / self.alpha
    }
}
