use super::*;

pub struct LinearWithWarmup {
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
        ensure!(
            warmup_steps <= max_steps,
            format!(
                "warmup_steps: {:?} exceeds max_steps: {:?}",
                warmup_steps, max_steps
            )
        );
        ensure!(
            max_steps > 0,
            format!("max_steps: {:?} must be positive", max_steps)
        );
        ensure!(
            base_lr > 0.0,
            format!("base_lr: {:?} must be positive", base_lr)
        );
        Ok(Self::new(warmup_steps, max_steps, base_lr))
    }
}

impl Schedule for LinearWithWarmup {
    fn get_lr(&self, step_t: u64) -> Result<f64> {
        ensure!(
            step_t <= self.max_steps,
            format!(
                "current step: {:?} exceeds max_steps: {:?}",
                step_t, self.max_steps
            )
        );
        if step_t <= self.warmup_steps {
            Ok(self.base_lr * (step_t as f64 / self.warmup_steps as f64))
        } else {
            Ok(self.base_lr * (self.max_steps as f64 - step_t as f64)
                / (self.max_steps as f64 - self.warmup_steps as f64))
        }
    }
}
