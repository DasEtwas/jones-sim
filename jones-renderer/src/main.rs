pub mod colormap;
pub mod state;

use std::f32::consts::TAU;
use crate::state::State;
use arc_swap::ArcSwap;
use jones_simulation::{Atom, Simulation};
use nalgebra::Vector2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::SurfaceError;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

#[tokio::main]
async fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).expect("failed to create window");

    let mut rng = StdRng::from_entropy();
    let mut rng2 = StdRng::from_entropy();
    let mut rng3 = StdRng::from_entropy();

    let temp = 3000.0;
    let side_length = 100;

    let hexagonal_lattice = |i: usize, rng: &mut StdRng, distance_factor: f32| -> Vector2<f32> {
        let n = (side_length as f32 / distance_factor).floor() as usize;
        Vector2::new(
            (i % n) as f32 + if (i / n) % 2 == 0 { 0.5 } else { 0.0 },
            (i / n) as f32 * 3.0f32.sqrt() * 0.5,
        ) * distance_factor
    };

    let rectangular_lattice = |i: usize, rng: &mut StdRng| -> Vector2<f32> {
        Vector2::new((i % side_length) as f32, (i / side_length) as f32)
    };

    let random = |i: usize, rng: &mut StdRng| -> Vector2<f32> {
        Vector2::new(
            rng.gen::<f32>() * side_length as f32,
            rng.gen::<f32>() * side_length as f32,
        )
    };

    // factor for inter atom distance
    let distance_factor = 0.99932;

    let count = side_length * side_length; // rect
    let count = (side_length as f32 / distance_factor).floor() as usize
        * (side_length as f32 / (3.0f32.sqrt() * 0.5) / distance_factor).floor() as usize; // hex

    //let count = side_length * side_length / 4; // random

    let vel = 0.0;

    let margin = 0.0;

    // https://www.mpie.de/4249939/grain-boundary-phase-transformation-liebscher

    let mut simulation = Simulation::new(
        (0..count)
            .map(|i| {
                let pos = hexagonal_lattice(i, &mut rng, distance_factor);
                let r = (rng2.gen::<f32>() * TAU).sin_cos();
                Atom::new(
                    pos + Vector2::repeat(margin * side_length as f32),
                    Vector2::new(r.0, r.1) * rng2.gen::<f32>().sqrt()
                        * temp,
                    //if pos.y > side_length as f32 * 0.5 {
                    //    Vector2::new(vel, -vel * 0.2)
                    //} else {
                    //    Vector2::new(-vel, vel * 0.2)
                    //},
                    // Vector2::new(
                    //     pos.y - side_length as f32 * 0.5,
                    //     -(pos.x - side_length as f32 * 0.5),
                    // ) * 300.0,
                    [0.7; 3],
                    1.0,
                )
            })
            .filter(|_| rng3.gen::<f32>() > 1e-2)
            .filter(|s| {
                let r = (s.pos - Vector2::repeat(side_length as f32 * (1.0 + 2.0 * margin) * 0.5))
                    .norm();
                r > side_length as f32 * 0.1 //&& r < side_length as f32 * 0.45
            }),
        side_length as f32,
        2.0,
        margin * 2.0,
        true,
    );

    let stars = Arc::new(ArcSwap::from_pointee(simulation.atoms.clone()));

    let mut state = State::new(&window, &simulation, stars.clone()).await;

    let tick_counter = Arc::new(AtomicU64::new(0));

    std::thread::spawn({
        let tick_counter = tick_counter.clone();
        let paused = state.paused.clone();
        move || {
            //std::thread::sleep(Duration::from_secs(5));

            let mut start = Instant::now();
            let mut tick = 0;
            loop {
                while paused.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                }

                // update simulation state

                simulation.update();
                tick += 1;
                tick_counter.fetch_add(1, Ordering::Relaxed);

                let now = Instant::now();
                let e = now.duration_since(start);
                if e > Duration::from_millis(340) {
                    println!(
                        "Avg step time {:.3?}, {:.0} Hz",
                        e / tick,
                        tick as f32 / 0.34
                    );
                    start = now;
                    tick = 0;
                }

                simulation.atoms.iter_mut().for_each(|a| {
                    let k = 0.01;
                    a.h = a.h * (1.0 - k) + a.force.norm() * k;
                });

                stars.store(Arc::new(simulation.atoms.clone()));
            }
        }
    });

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
        if window_id == window.id() && last.elapsed() > Duration::from_millis(15) =>
            {
                last = Instant::now();

                let tick = tick_counter.load(Ordering::Relaxed);

                let temp = state.update(tick);

                if state.paused.load(Ordering::Relaxed) {
                    window.set_title(&format!(
                        "Temperature: {:.3}, Rewind: {}, Paused ⏸ N<>M",
                        temp,
                        state.rewind.unwrap()
                    ));
                } else {
                    window.set_title(&format!("Temperature: {:.3}, Tick: {}", temp, tick));
                }

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
