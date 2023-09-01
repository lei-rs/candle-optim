use candle_core::{Result, Tensor};

pub(crate) fn clamp(x: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    let device = x.device();
    let dtype = x.dtype();
    let min = Tensor::from_slice(&[min], 1, device)?.to_dtype(dtype)?;
    let max = Tensor::from_slice(&[max], 1, device)?.to_dtype(dtype)?;
    let x = x.broadcast_minimum(&max)?.broadcast_maximum(&min)?;
    Ok(x)
}

pub(crate) fn norm_l2(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.sum_all()?.sqrt()
}
