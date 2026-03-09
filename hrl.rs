// ======================================================================
// ЭТАП 5: Hierarchical RL — MetaController + Skills
// Источник: FINAL_INTEGRATION_with_pfc.rs
// Ключевое отличие: exploration_temperature и impulse_control берутся
//   из CognitiveModulation (которая вычисляется из БОГАТОЙ нейрохимии),
//   а не из упрощённых f32 заглушек
// ======================================================================

use tch::{nn, nn::Module, Kind, Tensor};

use crate::neurochemistry::CognitiveModulation;

pub struct MetaController {
    pub network: nn::Sequential,
    pub num_skills: usize,
}

impl MetaController {
    pub fn new(vs: &nn::Path, state_dim: i64, hidden_dim: i64, num_skills: usize) -> Self {
        let network = nn::seq()
            .add(nn::linear(
                vs / "mc_fc1",
                state_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "mc_fc2",
                hidden_dim,
                num_skills as i64,
                Default::default(),
            ));

        MetaController { network, num_skills }
    }

    /// Выбор скилла. temperature из CognitiveModulation.exploration_temperature
    pub fn select_skill(&self, state_tensor: &Tensor, temperature: f32) -> usize {
        let logits = self.network.forward(state_tensor);
        let scaled = logits / f64::from(temperature);
        let probs = scaled.softmax(-1, Kind::Float);

        // Epsilon-greedy с temperature
        if rand::random::<f32>() < temperature * 0.1 {
            rand::random::<usize>() % self.num_skills
        } else {
            probs.argmax(-1, false).int64_value(&[]) as usize
        }
    }
}

pub struct Skill {
    pub id: usize,
    pub policy: nn::Sequential,
    pub termination_fn: nn::Sequential,
}

impl Skill {
    pub fn new(vs: &nn::Path, id: usize, state_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let skill_path = vs / format!("skill_{id}");

        let policy = nn::seq()
            .add(nn::linear(
                &skill_path / "pol_fc1",
                state_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &skill_path / "pol_fc2",
                hidden_dim,
                action_dim,
                Default::default(),
            ));

        let termination_fn = nn::seq()
            .add(nn::linear(
                &skill_path / "term_fc1",
                state_dim,
                32,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&skill_path / "term_fc2", 32, 1, Default::default()));

        Skill {
            id,
            policy,
            termination_fn,
        }
    }

    pub fn select_action(&self, state_tensor: &Tensor, temperature: f32) -> i64 {
        let logits = self.policy.forward(state_tensor);
        let scaled = logits / f64::from(temperature);
        scaled
            .softmax(-1, Kind::Float)
            .argmax(-1, false)
            .int64_value(&[])
    }

    /// Терминация скилла. impulse_control из CognitiveModulation
    pub fn should_terminate(&self, state_tensor: &Tensor, impulse_control: f32) -> bool {
        let logit = self.termination_fn.forward(state_tensor).squeeze();
        let prob_terminate = logit.sigmoid().double_value(&[]);
        let threshold = f64::from(1.0 - impulse_control);
        prob_terminate > threshold
    }
}

/// Иерархический RL: мета-контроллер выбирает скилл, скилл выбирает действие
pub struct HierarchicalRL {
    pub meta_controller: MetaController,
    pub skills: Vec<Skill>,
    pub current_skill: Option<usize>,
}

impl HierarchicalRL {
    pub fn new(
        vs: &nn::Path,
        state_dim: i64,
        action_dim: i64,
        hidden_dim: i64,
        num_skills: usize,
    ) -> Self {
        let hrl_path = vs / "hrl";

        let meta_controller =
            MetaController::new(&(&hrl_path / "meta"), state_dim, hidden_dim, num_skills);

        let skills: Vec<Skill> = (0..num_skills)
            .map(|i| Skill::new(&hrl_path, i, state_dim, action_dim, hidden_dim))
            .collect();

        HierarchicalRL {
            meta_controller,
            skills,
            current_skill: None,
        }
    }

    /// Выбор действия через HRL. Использует CognitiveModulation из РЕАЛЬНОЙ нейрохимии
    pub fn choose_action(&mut self, state_tensor: &Tensor, cog: &CognitiveModulation) -> i64 {
        let temperature = cog.exploration_temperature;
        let impulse_control = cog.impulse_control;

        // Проверка терминации текущего скилла
        if let Some(skill_id) = self.current_skill {
            if self.skills[skill_id].should_terminate(state_tensor, impulse_control) {
                self.current_skill = None;
            }
        }

        // Выбор нового скилла если нужно
        if self.current_skill.is_none() {
            let new_skill = self.meta_controller.select_skill(state_tensor, temperature);
            self.current_skill = Some(new_skill);
        }

        let active_skill = &self.skills[self.current_skill.unwrap()];
        active_skill.select_action(state_tensor, temperature)
    }
}
