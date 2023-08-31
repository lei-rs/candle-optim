use candle_core::Error;

mod optimizers;
mod ops;
mod schedulers;

#[macro_export]
macro_rules! assert{
    ($condition:expr, $message:expr) => {
        if !$condition {
            Error::msg($message)
        }
    };
}