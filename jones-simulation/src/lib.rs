use crate::hashgrid::HashGrid;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

pub mod hashgrid;

#[derive(Copy, Clone, Debug)]
pub struct Star {
    pub mass_point: MassData,
    pub vel: Vector2<f32>,
    pub force: Vector2<f32>,
    pub color: [f32; 3],
}

impl Star {
    pub const DENSITY: f32 = 250.0;

    pub fn new(pos: Vector2<f32>, vel: Vector2<f32>, color: [f32; 3], mass: f32) -> Self {
        Self {
            mass_point: MassData {
                position: pos,
                mass,
            },
            force: Vector2::zeros(),
            vel,
            color,
        }
    }

    pub fn radius(&self) -> f32 {
        (0.75 * self.mass_point.mass / (Self::DENSITY * std::f32::consts::PI)).cbrt()
    }

    pub fn color(&self) -> [f32; 3] {
        self.color
    }

    pub fn mass(&self) -> f32 {
        self.mass_point.mass
    }

    pub fn pos(&self) -> &Vector2<f32> {
        &self.mass_point.position
    }
}

/// Represents a mass point in space.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MassData {
    pub position: Vector2<f32>,
    pub mass: f32,
}

pub struct Simulation {
    pub stars: Vec<Star>,
    pub forces_buf: Vec<Vector2<f32>>,
    pub grid: HashGrid,
}

impl Simulation {
    pub fn new<I>(stars: I, side_length: f32, cell_size: f32, margin: f32) -> Self
    where
        I: IntoIterator<Item = Star>,
    {
        let stars: Vec<Star> = stars.into_iter().collect();
        Self {
            forces_buf: vec![Vector2::<f32>::zeros(); stars.len()],
            stars,
            grid: HashGrid::new(
                (side_length * (1.0 + margin) / cell_size) as usize,
                (side_length * (1.0 + margin) / cell_size) as usize,
                cell_size,
                false,
            ),
        }
    }

    pub fn update(&mut self) {
        let particles = self
            .stars
            .iter()
            .map(|star| (star.mass_point.position.x, star.mass_point.position.y, &()))
            .collect::<Vec<_>>();

        #[inline]
        fn interact(dx: f32, dy: f32, _: &(), _: &()) -> Vector2<f32> {
            let dist = (dx * dx + dy * dy) as f64;

            const DESIRED_RADIUS: f64 = 1.0;
            const SIGMA_FAC: f64 = 1.122462048309373; // 6th root of 2, the factor of the root relative to sigma
            const SIGMA: f64 = DESIRED_RADIUS / SIGMA_FAC;
            const SIGMA6: f64 = SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA; // precomputed sigma^6
            const E: f64 = 0.1;

            // dist normally has exponent 13, but using 14 we normalise the diff vector :^)
            // we multiply by 0.5 because we touch every particle twice in interaction (boo!)
            let f = (((24.0 * E * SIGMA6 * (dist.powi(3) - 2.0 * SIGMA6)) / dist.powi(7)) as f32)
                .max(-1e7)
                * 0.5;

            Vector2::new(f * dx, f * dy)
        }

        let grid = self.grid.populate(&particles);
        self.grid
            .interact(&particles, &mut self.forces_buf, &grid, interact);

        //let damping = 0.0005;
        //let damping = 0.000;
        let damping = 0.0001;

        self.stars
            .iter_mut()
            .zip(&mut self.forces_buf)
            .for_each(|(star, force)| {
                star.force = force.cast::<f32>();
                star.vel += star.force;
                star.vel *= 1.0 - damping;
                star.mass_point.position += 1e-6 * star.vel;

                if self.grid.periodic() {
                    star.mass_point.position.x = star
                        .mass_point
                        .position
                        .x
                        .rem_euclid(self.grid.size_x() as f32 * self.grid.cell_size());
                    star.mass_point.position.y = star
                        .mass_point
                        .position
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
