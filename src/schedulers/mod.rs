use candle_nn::optim::Optimizer;
use color_eyre::eyre::ensure;
use color_eyre::Result;

pub use constant_with_warmup::ConstantWithWarmup;
pub use cosine_with_warmup::CosineWithWarmup;
pub use linear_with_warmup::LinearWithWarmup;

mod constant_with_warmup;
mod cosine_with_warmup;
mod linear_with_warmup;

pub trait Schedule {
    fn get_lr(&self, step_t: u64) -> Result<f64>;
}

pub struct Scheduler<'a, S: Schedule> {
    schedule: &'a S,
    pub step_t: u64,
}