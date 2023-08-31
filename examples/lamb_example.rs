use color_eyre::Result;
use candle_core::{Device, DType, Module, Tensor};
use candle_nn::{Linear, linear, VarBuilder, VarMap};
use candle_optim::optimizers::{Lamb, Optimizer, ParamsLamb};
use candle_optim::schedulers::{ConstantWithWarmup, Scheduler};

fn gen_data() -> Result<(Tensor, Tensor)> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;
    Ok((sample_xs, sample_ys))
}

fn main() -> Result<()> {
    let (sample_xs, sample_ys) = gen_data()?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = linear(2, 1, vb.pp("linear"))?;
    let params = ParamsLamb::default();
    let schedule = ConstantWithWarmup::try_new(100, params.lr)?;
    let mut scheduler = Scheduler::new(&schedule);
    let mut opt = Lamb::new(varmap.all_vars(), params)?;

    for step in 0..10000 {
        let ys = model.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        scheduler.step(&mut opt)?;
        println!("{} {} {}", scheduler.step_t, loss.to_vec0::<f32>()?, scheduler.get_lr()?);
    }

    Ok(())
}