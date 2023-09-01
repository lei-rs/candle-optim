#[derive(Clone, Debug)]
pub struct ParamsSophia {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub rho: f64,
}

impl Default for ParamsSophia {
    fn default() -> Self {
        Self {
            lr: 0.0001,
            beta1: 0.965,
            beta2: 0.99,
            eps: 1e-6,
            weight_decay: 0.1,
            rho: 0.04,
        }
    }
}
