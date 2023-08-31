use candle_core::Result;

use crate::optimizers::Optimizer;
use crate::assert;

mod linear_with_warmup;
mod constant_with_warmup;

pub trait Schedule {
	fn get_lr(&self, step_t: u64) -> Result<f64>;
}

pub struct Scheduler<'a, S: Schedule> {
	schedule: &'a S,
	step_t: u64,
}

impl <'a, S: Schedule> Scheduler<'a, S> {
	pub fn new(schedule: S) -> Self {
		Self {
			schedule,
			step_t: 0,
		}
	}

	pub fn update_step(&mut self, step_t: u64) {
		self.step_t = step_t;
	}

	pub fn update_schedule(&mut self, schedule: S) {
		self.schedule = schedule;
	}

	pub fn step(&mut self, optimizer: &mut dyn Optimizer) -> Result<()> {
		self.step_t += 1;
		optimizer.set_lr(self.schedule.get_lr(self.step_t)?);
		Ok(())
	}

	pub fn inner(&self) -> &S {
		&self.schedule
	}

	pub fn inner_mut(&mut self) -> &mut S {
		&mut self.schedule
	}
}