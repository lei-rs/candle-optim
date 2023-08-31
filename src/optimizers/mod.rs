use color_eyre::Result;
use candle_core::{Module, Tensor, Var};
use candle_core::backprop::GradStore;

pub use adamw::{AdamW, ParamsAdamW};
pub use lamb::{Lamb, ParamsLamb};

use crate::ops::*;

mod adamw;
mod lamb;
mod sophia;

pub trait Optimizer {
	fn backward_step(&mut self, loss: &Tensor) -> Result<()>;
	fn get_lr(&self) -> f64;
	fn set_lr(&mut self, lr: f64);
}