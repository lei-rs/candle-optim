use super::*;

struct ConstantWithWarmup {
	warmup_steps: u64,
	base_lr: f64,
}

impl ConstantWithWarmup {
	fn new(warmup_steps: u64, base_lr: f64) -> Self {
		Self {
			warmup_steps,
			base_lr,
		}
	}

	pub fn try_new(warmup_steps: u64, base_lr: f64) -> Result<Self> {
		assert!(base_lr > 0.0, format!("base_lr: {:?} must be positive", base_lr));
		Ok(Self::new(warmup_steps, base_lr))
	}
}

impl Schedule for ConstantWithWarmup {
	fn get_lr(&self, step_t: u64) -> Result<f64> {
		if step_t <= self.warmup_steps {
			Ok(self.base_lr * (step_t / self.warmup_steps))
		} else {
			Ok(self.base_lr)
		}
	}
}