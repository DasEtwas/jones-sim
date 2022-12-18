use crate::hashgrid::HashGrid;
use nalgebra::{Vector2, Vector3};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use serde::{Deserialize, Serialize};

pub mod hashgrid;
pub mod tree;

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
    pub const SCALE: f32 = 5000.0;
    pub const THETA: f32 = 0.75;
    pub const GRAVITY: f32 = 1e-4;

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
            ),
        }
    }

    pub fn update(&mut self) {
        /*let mut tree = Node::new_root(-Vector2::repeat(Self::SCALE / 2.0), Self::SCALE);

        // insert stars into tree
        for star in &self.stars {
            if tree.contains(star.pos()) {
                tree.insert(&star.mass_point);
            }
        }

        let starslol = self.stars.clone();

        // calculate force on stars
        self.stars
            .par_iter_mut()
            .for_each(|star| {
                // let force = tree.force_on(&star.mass_point);

                let mut force = Vector2::zeros();
                for b in &starslol {
                    let diff = b.mass_point.position - star.mass_point.position;

                    const EPSILON: f32 = 0.05;
                    let dist = (EPSILON + diff.norm_squared()).sqrt();

                    const SIGMA: f32 = 1.0;
                    const E: f32 = 1.0 / 8.0;

                    force += diff
                        * (6.0 * SIGMA.powi(6) * (dist.powi(6) - 8.0 * E * SIGMA.powi(6)))
                        / dist.powi(14)
                }

                star.vel += force;
                star.force = force;

                // integration can be done here because tree doesn't change
                star.mass_point.position += 1e-6 * star.vel;
            });

        self.stars
            .iter_mut()
            .filter(|star| !tree.contains(star.pos()))
            .for_each(|star| star.mass_point.position = Vector2::from_element(f32::NAN))
            */

        let particles = self
            .stars
            .iter()
            .map(|star| (star.mass_point.position.x, star.mass_point.position.y, &()))
            .collect::<Vec<_>>();

        #[inline]
        fn interact((x1, y1, _): &(f32, f32, &()), (x2, y2, _): &(f32, f32, &())) -> Vector2<f32> {
            let diff_x = x2 - x1;
            let diff_y = y2 - y1;

            let dist = (diff_x * diff_x + diff_y * diff_y) as f64;

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

            Vector2::new(f * diff_x, f * diff_y)
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
                star.mass_point.position += 1e-5 * star.vel;
                *force = Vector2::zeros();
            });
    }
}

pub struct Galaxy {
    /// `stars[0]` is the center
    stars: Vec<Star>,
}

impl Galaxy {
    pub fn new(
        center: Star,
        num_stars: usize,
        radius: f32,
        mass_distribution: &MassDistribution,
        color: [f32; 3],
    ) -> Self {
        let mut rng = XorShiftRng::from_entropy();

        Self {
            stars: [center]
                .into_iter()
                .chain((0..num_stars).map(|_| {
                    let a = rng.gen::<f32>() * std::f32::consts::TAU;
                    let d = rng.gen::<f32>().sqrt() * radius;

                    let relative_pos = Vector2::new(a.sin(), a.cos()) * d;
                    let n = Vector3::cross(
                        &*Vector3::z_axis(),
                        &Vector3::new(relative_pos.x, relative_pos.y, 0.0),
                    );
                    let velocity = (Simulation::GRAVITY * center.mass() / d).sqrt();

                    Star::new(
                        center.pos() + relative_pos,
                        center.vel + n.xy().normalize() * velocity,
                        color,
                        1.0 + mass_distribution.sample(rng.gen()),
                    )
                }))
                .collect(),
        }
    }

    pub fn stars(&self) -> &Vec<Star> {
        &self.stars
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
