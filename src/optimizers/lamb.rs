use super::*;

/*
Citation:
@misc{you2020large,
      title={Large Batch Optimization for Deep Learning: Training BERT in 76 minutes},
      author={Yang You and Jing Li and Sashank Reddi and Jonathan Hseu and Sanjiv Kumar and Srinadh Bhojanapalli and Xiaodan Song and James Demmel and Kurt Keutzer and Cho-Jui Hsieh},
      year={2020},
      eprint={1904.00962},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
*/

#[derive(Clone, Debug)]
pub struct ParamsLamb {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsLamb {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarLamb {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct Lamb {
    vars: Vec<VarLamb>,
    step_t: usize,
    params: ParamsLamb,
}

impl Lamb {
    pub fn new(vars: Vec<Var>, params: ParamsLamb) -> Result<Self> {
        let vars = vars.into_iter().map(|var| {
            let first_moment = Var::zeros(
                var.shape(),
                var.dtype(),
                var.device()
            )?;
            let second_moment = Var::zeros(
                var.shape(),
                var.dtype(),
                var.device()
            )?;
            Ok(VarLamb {
                var,
                first_moment,
                second_moment
            })
        }).collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vars,
            params,
            step_t: 0
        })
    }

	pub fn new_lr(vars: Vec<Var>, lr: f64) -> Result<Self> {
		let params = ParamsLamb {
			lr,
			..Default::default()
		};
		Self::new(vars, params)
	}

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;

        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;

        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));

        for var in self.vars.iter_mut() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;

            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let v_sqrt = v_hat.sqrt()?;

                let mut r = (m_hat / &(v_sqrt + self.params.eps)?)?;
                if self.params.weight_decay > 0.0 {
                    r = (r + (theta.as_tensor() * lr_lambda)?)?;
                }

                let w_norm = norm_l2(theta)?;
                let g_norm = norm_l2(&r)?;
                let ratio = clamp(&(w_norm / g_norm)?, 0.08, 0.5)?; // clamp values from deepspeed
                let next_theta = (theta.as_tensor() - (ratio.broadcast_mul(&r)? * lr)?)?;

                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl Optimizer for Lamb {
	fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
		self.step(&loss.backward()?)
	}

    fn get_lr(&self) -> f64 {
        self.params.lr
    }

	fn set_lr(&mut self, lr: f64) {
		self.params.lr = lr;
	}
}