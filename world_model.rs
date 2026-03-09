// ======================================================================
// ЭТАП 6: World Model — VAE с GRU-динамикой
// Источник: FINAL_INTEGRATION_with_pfc.rs
// Назначение: предсказание следующего состояния мира
// Интеграция: state_dim=10 (pos + нейрохимия + гомеостаз)
// ======================================================================

use tch::{nn, nn::Module, Tensor};

pub struct WorldModel {
    pub encoder: nn::Sequential,
    pub decoder: nn::Sequential,
    pub dynamics: nn::GRU,
}

impl WorldModel {
    pub fn new(vs: &nn::Path, state_dim: i64, hidden_dim: i64) -> Self {
        let wm_path = vs / "world_model";

        let encoder = nn::seq()
            .add(nn::linear(
                &wm_path / "enc1",
                state_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &wm_path / "enc2",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ));

        let decoder = nn::seq()
            .add(nn::linear(
                &wm_path / "dec1",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &wm_path / "dec2",
                hidden_dim,
                state_dim,
                Default::default(),
            ));

        let dynamics = nn::gru(
            &wm_path / "dynamics",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );

        WorldModel {
            encoder,
            decoder,
            dynamics,
        }
    }

    /// Кодирование состояния в латентное пространство
    pub fn encode(&self, state: &Tensor) -> Tensor {
        self.encoder.forward(state)
    }

    /// Декодирование из латентного пространства
    pub fn decode(&self, latent: &Tensor) -> Tensor {
        self.decoder.forward(latent)
    }

    /// Предсказание следующего состояния
    pub fn predict_next_state(&self, state: &Tensor) -> Tensor {
        let latent = self.encode(state);
        self.decode(&latent)
    }
}
