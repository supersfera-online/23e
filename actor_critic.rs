// ======================================================================
// ЭТАП 4: Actor-Critic — нейросетевой RL
// Источник: main_with_tom.rs (богатая версия с 3-слойной сетью)
// Сохранено: temperature-based exploration, TD-error update
// Интеграция: state_dim=10 (pos + нейрохимия + гомеостаз),
//   learning_rate_multiplier и temporal_discount из CognitiveModulation
// ======================================================================

use tch::{nn, nn::Module, nn::OptimizerConfig, Kind, Tensor};

pub struct ActorCritic {
    pub actor: nn::Sequential,
    pub critic: nn::Sequential,
}

impl ActorCritic {
    pub fn new(vs: &nn::Path, state_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let actor = nn::seq()
            .add(nn::linear(
                vs / "actor_fc1",
                state_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "actor_fc2",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "actor_fc3",
                hidden_dim / 2,
                action_dim,
                Default::default(),
            ));

        let critic = nn::seq()
            .add(nn::linear(
                vs / "critic_fc1",
                state_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "critic_fc2",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "critic_fc3",
                hidden_dim / 2,
                1,
                Default::default(),
            ));

        ActorCritic { actor, critic }
    }

    /// Выбор действия с temperature из CognitiveModulation.exploration_temperature
    pub fn select_action(&self, state_tensor: &Tensor, temperature: f32) -> (i64, Tensor) {
        let logits = self.actor.forward(state_tensor) / f64::from(temperature);
        let probs = logits.softmax(-1, Kind::Float);
        let action = probs.multinomial(1, true);
        let action_idx = action.int64_value(&[0, 0]);
        (action_idx, probs)
    }

    /// Оценка состояния критиком
    pub fn get_value(&self, state_tensor: &Tensor) -> f32 {
        let value = self.critic.forward(state_tensor);
        value.double_value(&[0, 0]) as f32
    }

    /// Пакетное обновление с параметрами из CognitiveModulation
    /// discount = cog.temporal_discount, lr_mult = cog.learning_rate_multiplier
    pub fn batch_update(
        &self,
        optimizer: &mut nn::Optimizer,
        states: &Tensor,
        actions: &Tensor,
        rewards: &Tensor,
        next_states: &Tensor,
        dones: &Tensor,
        discount: f32,
    ) {
        let values = self.critic.forward(states);
        let next_values = self.critic.forward(next_states).detach();

        let targets = rewards + f64::from(discount) * &next_values * (1.0 - dones);
        let td_errors = &targets - &values;

        // Critic loss: MSE of TD errors
        let critic_loss = td_errors.pow_tensor_scalar(2).mean(Kind::Float);

        // Actor loss: policy gradient with advantages
        let logits = self.actor.forward(states);
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let action_log_probs = log_probs.gather(1, actions, false);
        let advantages = td_errors.detach();
        let actor_loss = -(action_log_probs * advantages).mean(Kind::Float);

        let total_loss = &actor_loss + &critic_loss;

        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();
    }
}
