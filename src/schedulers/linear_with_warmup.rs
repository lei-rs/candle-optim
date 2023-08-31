use super::*;

struct LinearWithWarmup {
	warmup_steps: u64,
	max_steps: u64,
	base_lr: f64,
}

impl LinearWithWarmup {
	fn new(warmup_steps: u64, max_steps: u64, base_lr: f64) -> Self {
		Self {
			warmup_steps,
			max_steps,
			base_lr,
		}
	}

	pub fn try_new(warmup_steps: u64, max_steps: u64, base_lr: f64) -> Result<Self> {
		assert!(warmup_steps <= max_steps, format!("warmup_steps: {:?} exceeds max_steps: {:?}", warmup_steps, max_steps));
		assert!(max_steps > 0, format!("max_steps: {:?} must be positive", max_steps));
		assert!(base_lr > 0.0, format!("base_lr: {:?} must be positive", base_lr));
		Ok(Self::new(warmup_steps, max_steps, base_lr))
	}

	pub fn last_lr(&self) -> Result<f64> {
		self.get_lr(self.max_steps)
	}
}

impl Schedule for LinearWithWarmup {
	fn get_lr(&self, step_t: u64) -> Result<f64> {
		assert!(step_t <= self.max_steps, format!("current step: {:?} exceeds max_steps: {:?}", step_t, self.max_steps));
		if step_t <= self.warmup_steps {
			Ok(self.base_lr * (step_t / self.warmup_steps))
		} else {
			Ok(self.base_lr * (self.max_steps - step_t) / (self.max_steps - self.warmup_steps))
		}
	}
}