use super::*;

/*
Original code from https://github.com/huggingface/candle/blob/main/candle-nn/src/optim.rs
This implementation supports amsgrad.

Citation(s):
@misc{loshchilov2019decoupled,
      title={Decoupled Weight Decay Regularization},
      author={Ilya Loshchilov and Frank Hutter},
      year={2019},
      eprint={1711.05101},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{reddi2019convergence,
      title={On the Convergence of Adam and Beyond},
      author={Sashank J. Reddi and Satyen Kale and Sanjiv Kumar},
      year={2019},
      eprint={1904.09237},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
*/

#[derive(Clone, Debug)]
pub struct ConfigAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
}

impl Default for ConfigAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
    vhm: Option<Var>,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ConfigAdamW,
}

impl Optimizer for AdamW {
    type Config = ConfigAdamW;

    fn new(vars: Vec<Var>, config: Self::Config) -> candle_core::Result<Self> {
        let vars = vars
            .into_iter()
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                let vhm = match config.amsgrad {
                    true => Some(Var::zeros(shape, dtype, device)?),
                    false => None,
                };
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                    vhm,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params: config,
        })
    }

    fn step(&mut self, grads: &GradStore) -> candle_core::Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;

        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));

        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;

            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;

                // vhm update
                if self.step_t == 1 && self.params.amsgrad {
                    &var.vhm.unwrap().set(&v_hat)?;
                } else if self.params.amsgrad {
                    let vhm = &var.vhm.unwrap();
                    vhm.set(&v_hat.broadcast_maximum(vhm)?)?;
                }

                let v_hat = match self.params.amsgrad {
                    true => var.vhm.as_ref().unwrap().as_tensor(),
                    false => &v_hat,
                };

                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;

                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}
