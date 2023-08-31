use candle_core::{Result};
use crate::optimizers::Optimizer;

pub trait Schedule {
	fn get_lr(&self, step: u64) -> f64;
}

pub struct Scheduler<'a, O: Optimizer, S: Schedule> {
	optimizer: &'a mut O,
	lr_schedule: &'a S,
	step: u64,
}

impl <'a, O: Optimizer, S: Schedule> Scheduler<'a, O, S> {
	pub fn new(optimizer: &'a mut O, lr_schedule: S) -> Self {
		Self {
			optimizer,
			lr_schedule,
			step: 0,
		}
	}

	pub fn step(&mut self) -> Result<()> {
		self.optimizer.set_lr(self.lr_schedule.get_lr(self.step));
		self.step += 1;
		Ok(())
	}
}