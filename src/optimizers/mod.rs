use candle_core::{Module, Result, Tensor, Var};
use candle_core::backprop::GradStore;

use crate::ops::*;

mod lamb;
mod adamw;
use lamb::Lamb;
use adamw::AdamW;

pub trait Optimizer {
	fn backward_step(&mut self, loss: &Tensor) -> Result<()>;
	fn get_lr(&self) -> f64;
	fn set_lr(&mut self, lr: f64);
}