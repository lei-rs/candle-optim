use color_eyre::Result;
use color_eyre::eyre::{ensure};
pub use constant_with_warmup::ConstantWithWarmup;
pub use cosine_with_warmup::CosineWithWarmup;
pub use linear_with_warmup::LinearWithWarmup;

use crate::optimizers::Optimizer;

mod linear_with_warmup;
mod constant_with_warmup;
mod cosine_with_warmup;

pub trait Schedule {
	fn get_lr(&self, step_t: u64) -> Result<f64>;
}

pub struct Scheduler<'a, S: Schedule> {
	schedule: &'a S,
	pub step_t: u64,
}

impl <'a, S: Schedule> Scheduler<'a, S> {
	pub fn new(schedule: &'a S) -> Self {
		Self {
			schedule,
			step_t: 0,
		}
	}

	pub fn update_step(&mut self, step_t: u64) {
		self.step_t = step_t;
	}

	pub fn update_schedule(&mut self, schedule: &'a S) {
		self.schedule = schedule;
	}

	pub fn get_lr(&self) -> Result<f64> {
		self.schedule.get_lr(self.step_t)
	}

	pub fn step(&mut self, optimizer: &mut dyn Optimizer) -> Result<()> {
		self.step_t += 1;
		optimizer.set_lr(self.get_lr()?);
		Ok(())
	}

	pub fn inner(&self) -> &S {
		&self.schedule
	}
}