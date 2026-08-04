# Event Camera Control for an Inverted Pendulum

This project simulates closed-loop control of an inverted pendulum on a cart
using event-camera signals. It combines:

- an inverted-pendulum/cart physics simulator,
- an IEBCS-style Dynamic Vision Sensor (DVS) event generator,
- a transmission channel for bandwidth, latency, jitter, and packet loss,
- event filtering with a cart-centered pendulum ROI,
- event-based state estimation, and
- LQR feedback control.

The intended pipeline is:

```text
Pendulum renderer
  -> DVS sensor
  -> transmission channel
  -> ROI / polarity / refractory filter
  -> event-based state estimator
  -> LQR controller
  -> pendulum dynamics
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you already use the checked-in local virtual environment, run commands with:

```bash
.venv\Scripts\python.exe ...
```

## Run

Headless batch run:

```bash
python example\closed_loop_system.py --headless --duration 20 --estimator event
```

Interactive run with the pendulum display:

```bash
python example\closed_loop_system.py --duration 20 --estimator event
```

Debug run with noisy ground-truth states instead of the event estimator:

```bash
python example\closed_loop_system.py --headless --duration 10 --estimator ground_truth
```

Optional event-stream display:

```bash
python example\closed_loop_system.py --display-events
```

Simulation results are saved to:

```text
outputs/closed_loop_event_lqr_result.npz
```

## Project Structure

```text
README.md
requirements.txt
src/
  dvs_sensor.py              IEBCS-style DVS sensor model
  event_buffer.py            DVS event container
  event_display.py           OpenCV event display
  dat_files.py               .dat event file I/O
example/
  closed_loop_system.py      Main closed-loop experiment
  integrated_event_camera.py DVS + transmission + filter wrapper
  event_transmission.py      Bandwidth/loss/latency/jitter channel
  event_filter.py            ROI, polarity, refractory, and noise filters
  event_based_estimator.py   Event-line-fit four-state estimator
  inverted_pendulum_simulator.py
  pendulum_controller.py
  simple_state_estimator.py  Noisy ground-truth debug estimator
```

## Reference Projects

- Event-camera simulation is based on the local `IEBCS` project and its DVS
  sensor/event-buffer implementation.
- Cart-pendulum dynamics and LQR feedback are informed by the local
  `inverted_pendulum` project.

## Current Notes

The default experiment uses real event-derived state estimates, not ground
truth. The estimator fits the pendulum rod from filtered event points and
derives angle, angular velocity, cart position, and cart velocity. The
ground-truth estimator remains available only for controller debugging.
