// ======================================================================
// ЭТАП 5: Meta-Learner (RL²) — GRU-based мета-обучение
// Источник: FINAL_INTEGRATION_with_pfc.rs
// Сохранено: recurrent policy с контекстом (state + last_action + last_reward)
// Интеграция: hidden_dim совпадает с остальными модулями
// ======================================================================

use tch::{nn, nn::Module, nn::RNN, Device, Kind, Tensor};

use crate::world::Action;

pub struct MetaLearner {
    pub gru: nn::GRU,
    pub policy_head: nn::Sequential,
    pub value_head: nn::Sequential,
    pub hidden_state: Tensor,
    hidden_dim: i64,
}

impl MetaLearner {
    pub fn new(vs: &nn::Path, state_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let ml_path = vs / "meta_learner";

        // Вход GRU: state + one-hot action + reward
        let input_size = state_dim + action_dim + 1;

        let gru = nn::gru(
            &ml_path / "gru",
            input_size,
            hidden_dim,
            Default::default(),
        );

        let policy_head = nn::seq().add(nn::linear(
            &ml_path / "pol_h",
            hidden_dim,
            action_dim,
            Default::default(),
        ));

        let value_head = nn::seq().add(nn::linear(
            &ml_path / "val_h",
            hidden_dim,
            1,
            Default::default(),
        ));

        let hidden_state = Tensor::zeros([1, 1, hidden_dim], (Kind::Float, Device::Cpu));

        MetaLearner {
            gru,
            policy_head,
            value_head,
            hidden_state,
            hidden_dim,
        }
    }

    /// Выбор действия с учётом предыдущего контекста (RL²)
    pub fn select_action(
        &mut self,
        state_tensor: &Tensor,
        last_action: i64,
        last_reward: f32,
    ) -> i64 {
        // One-hot кодирование прошлого действия
        let action_tensor = Tensor::from_slice(&[last_action]);
        let one_hot = action_tensor
            .one_hot(Action::ACTION_DIM)
            .to_kind(Kind::Float)
            .unsqueeze(0);

        let reward_tensor = Tensor::from_slice(&[last_reward]).unsqueeze(0);

        // Конкатенация: [state, one_hot_action, reward]
        let input = Tensor::cat(&[state_tensor.shallow_clone(), one_hot, reward_tensor], -1);

        // GRU step: input shape [seq_len=1, batch=1, features]
        let gru_state = self.gru.step(&input.unsqueeze(0), &self.hidden_state);
        self.hidden_state = gru_state.h().shallow_clone();

        // Policy head
        let h = self.hidden_state.squeeze_dim(0);
        let logits = self.policy_head.forward(&h);
        logits
            .softmax(-1, Kind::Float)
            .argmax(-1, false)
            .int64_value(&[])
    }

    pub fn get_value(&self) -> f32 {
        let h = self.hidden_state.squeeze_dim(0);
        let value = self.value_head.forward(&h);
        value.double_value(&[0, 0]) as f32
    }

    pub fn reset_hidden_state(&mut self) {
        self.hidden_state = Tensor::zeros(
            [1, 1, self.hidden_dim],
            (Kind::Float, Device::Cpu),
        );
    }
}
