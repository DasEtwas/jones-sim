pub mod colormap;
pub mod state;

use crate::state::State;
use gravsim_simulation::{MassDistribution, Simulation, Star};
use nalgebra::Vector2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};
use wgpu::SurfaceError;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

#[tokio::main]
async fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).expect("failed to create window");

    let mass_distribution = MassDistribution::new(35.0, 200.0);
    let mut rng = StdRng::from_entropy();
    let mut rng2 = StdRng::from_entropy();
    let mut rng3 = StdRng::from_entropy();

    let temp = 0.0;
    let side_length = 300;

    let hexagonal_lattice = |i: usize, rng: &mut StdRng| -> Vector2<f32> {
        Vector2::new(
            (i % side_length) as f32 + (rng.gen::<f32>() * 2.0 - 1.0) * 1e-2,
            (i / side_length) as f32
                + (rng.gen::<f32>() * 2.0 - 1.0) * 1e-2
                + 0.5 * if i % 2 == 0 { 1.0 } else { 0.0 },
        )
        .component_mul(&Vector2::new(3.0f32.sqrt() * 0.5, 1.0))
    };

    let random = |i: usize, rng: &mut StdRng| -> Vector2<f32> {
        Vector2::new(
            rng.gen::<f32>() * side_length as f32,
            rng.gen::<f32>() * side_length as f32,
        )
    };

    let count = side_length * side_length;
    //let count = side_length * side_length / 4;

    let simulation = Simulation::new(
        (0..count)
            .map(|i| {
                Star::new(
                    hexagonal_lattice(i, &mut rng),
                    //random(i, &mut rng),
                    Vector2::new(rng2.gen::<f32>() * 2.0 - 1.0, rng2.gen::<f32>() * 2.0 - 1.0)
                        * temp,
                    [0.7; 3],
                    50.0,
                )
            })
            .filter(|_| rng3.gen::<f32>() > 1e-2),
        side_length,
        2.0,
    );

    let mut state = State::new(&window, simulation).await;
    let mut last = Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            window_id,
            ref event,
        } if window_id == window.id() && !state.input(event) => match event {
            WindowEvent::Resized(new_size) => state.resize(*new_size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size)
            }
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        },
        Event::MainEventsCleared => window.request_redraw(),
        Event::RedrawRequested(window_id)
            if window_id == window.id() && last.elapsed() > Duration::from_millis(30) =>
        {
            state.update();
            last = Instant::now();

            match state.render() {
                Ok(_) => {}
                Err(e) => match e {
                    SurfaceError::OutOfMemory => *control_flow = ControlFlow::Exit,
                    SurfaceError::Lost => state.resize(state.size),
                    _ => eprintln!("Render Error: {:?}", e),
                },
            }
        }
        _ => {}
    });
}
