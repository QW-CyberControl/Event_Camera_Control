## Project Overview

This project is a **simulator for a closed‑loop control system of an inverted pendulum driven by an event camera**. It integrates a physics engine (inverted pendulum dynamics), an event camera simulator, a state estimator, and a controller to investigate the potential of event‑based vision in high‑speed control loops. By emulating the asynchronous output of an event camera, the system estimates the pendulum angle in real time, computes the control force, and stabilizes the pendulum. The simulator provides visualisation, data logging, and performance analysis tools to facilitate algorithm debugging and performance evaluation.

## Running the Simulation

1. Clone or download the project code.

2. Navigate to the `example/` directory.

3. Run the main script:

```
python closed_loop_system.py
```

## Project Structure

├── README.md                                                 # This file
├── example/                                                      # Main application modules
│   ├── closed_loop_system.py                         # Main closed‑loop orchestrator
│   ├── integrated_event_camera.py               # Event camera simulator (wrapper for src/dvs_sensor)
│   ├── inverted_pendulum_simulator.py       # Inverted pendulum physics simulator
│   ├── pendulum_controller.py                       # Controller (PID, LQR)
│   ├── realtime_monitor.py                             # Real‑time monitoring GUI (optional)
│   ├── simple_state_estimator.py                   # Simplified state estimator (with noise/delay)
│   └── outputs/                                                  # Output directory (created automatically)
│       ├── simulation_data_*.npz                        # (commented out) Logged data
│       ├── simulation_report_*.png                      # (commented out) Generated charts
│       └── simulation_report_*.txt                    # (commented out) Text report
└── src/                                                                # Low‑level event camera simulation modules
    ├── dvs_sensor.py                                         # DVS sensor model (ICNS)
    ├── event_buffer.py                                      # Event storage and manipulation
    ├── event_display.py                                    # Real‑time event visualisation
    └── dat_files.py                                             # Read/write .dat event files

## Module Descriptions

### `example/` – Application Modules

#### `inverted_pendulum_simulator.py`

Implements the physics of an inverted pendulum on a cart.

#### `integrated_event_camera.py`

Wraps the low‑level DVS sensor (`src/dvs_sensor.py`) to interface with the pendulum simulator. It converts pendulum images into event streams, maintains event statistics, and displays the event output. Configurable parameters include event thresholds, noise, etc.

#### `simple_state_estimator.py`

Estimates the pendulum angle and angular velocity from the event stream.  
**Current status:** This module is a placeholder and does **not** yet perform accurate estimation.

#### `pendulum_controller.py`

Computes the control force based on the estimated state. Supports multiple controller types (PID, LQR).  
**Current status:** Although the controller can compute a force, the system is **not yet stable**. The gains are only roughly tuned, and the estimator inaccuracy prevents proper stabilisation.

#### `closed_loop_system.py`

Orchestrates the entire simulation:

- Initialises all modules (pendulum, camera, estimator, controller).

- Runs the main loop at the event camera rate.

- For each camera frame:
  
  1. Renders the current pendulum image.
  
  2. Feeds it to the event camera to generate events.
  
  3. Passes events to the estimator.
  
  4. Computes the control force.
  
  5. Applies the force to the pendulum (multiple physics substeps).
  
  6. Updates the display and performance statistics.

### `src/` – Low‑Level Event Camera Modules

The following four modules are based on the [IEBCS project](https://github.com/neuromorphicsystems/IEBCS) and provide a realistic DVS simulation.

#### `dvs_sensor.py`

Implements a realistic Dynamic Vision Sensor (DVS) model based on the **ICNS**. 

The `update()` method processes a new image and returns an `EventBuffer` containing all generated events.

#### `event_buffer.py`

Provides a flexible container for DVS events. Supports adding, removing, merging, sorting, and writing events to `.dat` files. Used by the sensor and display modules.

#### `event_display.py`

Creates an OpenCV window to visualise events in real time.

#### `dat_files.py`

Utility functions to read and write `.dat` event files (a common format for DVS data). Used by `event_buffer.py` for file I/O.

## Core Class Documentation

### `InvertedPendulumSimulator`

- **`__init__(config)`**: Initialises physical parameters and visual settings.

- **`step(force)`**: Advances simulation by one time step (RK4).

- **`get_current_image()`**: Renders the current state as a BGR image.

- **`reset()`**: Resets the pendulum to its initial state.

### `IntegratedEventCamera`

- **`__init__(width, height, config)`**: Creates a DVS sensor with the given resolution and configuration.

- **`init_with_frame(frame)`**: Initialises the sensor with the first image.

- **`process_frame(frame, dt_us)`**: Generates events from the current frame and returns an `EventBuffer`.

- **`get_event_statistics()`**: Returns total events, average rate, etc.

- **`reset()`**: Resets the camera.

### `SimpleStateEstimator`

- **`__init__(width, height, config)`**: Configures noise levels, delay, history size.

- **`estimate_from_events(events, current_time_us)`**: Returns estimated angle and angular velocity.

- **`set_ground_truth_callback(callback)`**: (Debug) Provides a callback to obtain ground‑truth states.

- **`reset()`**: Clears internal buffers.

### `PendulumController`

- **`__init__(config)`**: Sets controller type, gains, limits.

- **`compute_control(angle, angular_velocity, current_time)`**: Computes control force.

- **`reset()`**: Resets integral error and histories.

- **`get_control_statistics()`**: Returns average force, max force, RMS error.

### `ClosedLoopSystem`

- **`__init__(config)`**: Instantiates all sub‑modules and prepares output directory.

- **`run_simulation()`**: Executes the main closed‑loop simulation.

- **`_display_current_state()`**: Shows the pendulum image with overlaid information.

- **`_update_performance_stats()`**: Updates loop timing statistics.

- **`_print_performance_stats()`**: Prints summary after simulation.

## Current Status & Future Work

- **State estimation:** The `SimpleStateEstimator` is a placeholder; accurate estimation from event data is not yet implemented. A robust method is needed.

- **Control:** The controller gains are not tuned for stability, and the estimator errors prevent the system from balancing. Either a more sophisticated estimator or an adaptive/robust controller is required.


