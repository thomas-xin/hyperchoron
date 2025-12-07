use midly::{Smf, Timing, TrackEventKind, MetaMessage};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;


#[pyfunction]
fn parse_midi_events<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray2<i32>>> {
	let bytes = fs::read(path)?;
	let smf = Smf::parse(&bytes)
		.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

	// [time, event_type, channel, pitch/data1, velocity/data2, duration/data3]
	let mut events: Vec<[i32; 6]> = Vec::new();

	// Active notes: key = (channel, pitch), value = stack of (event_index, start_time, velocity)
	let mut active: HashMap<(u8, u8), Vec<(usize, u32, u8)>> = HashMap::new();

	match smf.header.timing {
		Timing::Timecode(fps, sub) => {
			events.push([
				0,
				0xFF,
				smf.header.format as i32,
				1,
				sub.into(),
				0,
			]);
			events.push([
				0,
				0x51,
				(1000000. / fps.as_f32()) as i32,
				0,
				0,
				0,
			]);
		}
		Timing::Metrical(tpb) => {
			events.push([
				0,
				0xFF,
				smf.header.format as i32,
				1,
				tpb.as_int() as i32,
				0,
			]);
		}
	}

	for track in smf.tracks.iter() {
		let mut abs_time: u32 = 0;
		for event in track {
			abs_time += event.delta.as_int();

			match &event.kind {
				// -----------------------------
				// Channel Voice Messages
				// -----------------------------
				TrackEventKind::Midi { channel, message } => {
					let ch = channel.as_int() as i32;
					match message {
						midly::MidiMessage::NoteOn { key, vel } => {
							let pitch = key.as_int();
							let v = vel.as_int();

							if v > 0 {
								// New NoteOn: store with zero duration for now
								let idx = events.len();
								events.push([
									abs_time as i32,
									0x90,
									ch,
									pitch as i32,
									v as i32,
									0,
								]);
								active.entry((ch as u8, pitch))
									.or_default()
									.push((idx, abs_time, v));
							} else {
								// Velocity 0 => NoteOff
								if let Some(stack) = active.get_mut(&(ch as u8, pitch)) {
									if let Some((idx, start_time, _vel)) = stack.pop() {
										let dur = abs_time - start_time;
										events[idx][5] = dur as i32;
									}
								}
								// Do not output NoteOffs
							}
						}
						midly::MidiMessage::NoteOff { key, vel: _ } => {
							let pitch = key.as_int();
							if let Some(stack) = active.get_mut(&(ch as u8, pitch)) {
								if let Some((idx, start_time, _vel)) = stack.pop() {
									let dur = abs_time - start_time;
									events[idx][5] = dur as i32;
								}
							}
							// Do not output NoteOffs
						}
						midly::MidiMessage::Controller { controller, value } => {
							events.push([
								abs_time as i32,
								0xB0,
								ch,
								controller.as_int() as i32,
								value.as_int() as i32,
								0,
							]);
						}
						midly::MidiMessage::ProgramChange { program } => {
							events.push([
								abs_time as i32,
								0xC0,
								ch,
								program.as_int() as i32,
								0,
								0,
							]);
						}
						midly::MidiMessage::PitchBend { bend } => {
							events.push([
								abs_time as i32,
								0xE0,
								ch,
								(bend.as_int() as i32) + 0x2000,
								0,
								0,
							]);
						}
						_ => {}
					}
				}
				// -----------------------------
				// Meta Events
				// -----------------------------
				TrackEventKind::Meta(meta) => match meta {
					MetaMessage::Tempo(t) => {
						events.push([
							abs_time as i32,
							0x51,
							t.as_int() as i32,
							0,
							0,
							0,
						]);
					}
					MetaMessage::TimeSignature(numer, denom, _, _) => {
						events.push([
							abs_time as i32,
							0x58,
							*numer as i32,
							(1 << *denom) as i32,
							0,
							0,
						]);
					}
					MetaMessage::EndOfTrack => {
						events.push([
							abs_time as i32,
							0x2F,
							0,
							0,
							0,
							0,
						]);
					}
					_ => {}
				},
				_ => {}
			}
		}
	}

	let n = events.len();
	// Allocate a new 2D NumPy array (shape: n x 6). Using new() for broader version compatibility.
	let arr = unsafe { PyArray2::<i32>::new(py, [n, 6], false) };

	unsafe {
		let mut view = arr.as_array_mut();
		for (i, ev) in events.into_iter().enumerate() {
			view[[i, 0]] = ev[0];
			view[[i, 1]] = ev[1];
			view[[i, 2]] = ev[2];
			view[[i, 3]] = ev[3];
			view[[i, 4]] = ev[4];
			view[[i, 5]] = ev[5];
		}
	}

	Ok(arr)
}

#[pymodule]
fn fastmidi(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(parse_midi_events, m)?)?;
	Ok(())
}
