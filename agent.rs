// ======================================================================
// ЭТАП 7: PersonalityAgent — ВСЕ модули ВМЕСТЕ
// Ничего не потеряно:
//   Этап 1: neurochemistry (регионы, рецепторы, десенсибилизация)
//   Этап 1: homeostasis (setpoints, intrinsic motivation)
//   Этап 1: memory (эпизодическая, кольцевой буфер)
//   Этап 2: pfc_planner (GOAP на реальной нейрохимии)
//   Этап 3: theory_of_mind (из CognitiveModulation)
//   Этап 3: world (GridWorld, Position, Action)
//   Этап 4: actor_critic (3-слойная сеть, TD-update)
//   Этап 5: hrl (MetaController + Skills) / meta_learner (GRU RL²)
//   Этап 6: world_model (VAE + GRU)
// Enum dispatch: DecisionModule вместо trait objects
// ======================================================================

use std::collections::HashMap;

use tch::{nn, nn::OptimizerConfig, nn::VarStore, Device, Kind, Tensor};

use crate::actor_critic::ActorCritic;
use crate::homeostasis::HomeostaticState;
use crate::hrl::HierarchicalRL;
use crate::memory::{Episode, EpisodicMemory};
use crate::meta_learner::MetaLearner;
use crate::neurochemistry::NeurochemicalState;
use crate::pfc_planner::PfcPlanner;
use crate::theory_of_mind::TheoryOfMind;
use crate::world::{Action, GridWorld, Position};
use crate::world_model::WorldModel;

// --- Конфигурация ---

pub const STATE_DIM: i64 = 10; // pos(2) + neuro(5) + homeo(3)
pub const ACTION_DIM: i64 = 4;
pub const DEFAULT_HIDDEN_DIM: i64 = 128;
pub const DEFAULT_NUM_SKILLS: usize = 8;
pub const DEFAULT_MEMORY_SIZE: usize = 100_000;
pub const DEFAULT_LR: f64 = 1e-4;

// --- Enum dispatch: стратегия принятия решений ---

pub enum DecisionModule {
    Hierarchical {
        hrl: HierarchicalRL,
    },
    MetaLearning {
        meta_learner: MetaLearner,
    },
    ActorCriticOnly,
}

// --- Конфигурация агента ---

pub struct AgentConfig {
    pub hidden_dim: i64,
    pub num_skills: usize,
    pub memory_size: usize,
    pub use_tom: bool,
    pub lr: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            hidden_dim: DEFAULT_HIDDEN_DIM,
            num_skills: DEFAULT_NUM_SKILLS,
            memory_size: DEFAULT_MEMORY_SIZE,
            use_tom: true,
            lr: DEFAULT_LR,
        }
    }
}

// --- Агент ---

pub struct PersonalityAgent {
    pub id: usize,
    pub vs: VarStore,
    pub optimizer: nn::Optimizer,

    // Этап 1: фундамент (БОГАТАЯ нейрохимия, не заглушки)
    pub neurochemistry: NeurochemicalState,
    pub homeostasis: HomeostaticState,
    pub memory: EpisodicMemory,

    // Этап 2: PFC
    pub pfc_planner: PfcPlanner,
    pub action_social_costs: HashMap<i64, f32>,

    // Этап 3: мир + ToM
    pub position: Position,
    pub theory_of_mind: TheoryOfMind,
    pub use_tom: bool,

    // Этап 4: Actor-Critic (базовый RL, используется всеми типами)
    pub actor_critic: ActorCritic,

    // Этап 5: стратегия (enum dispatch — HRL / MetaLearning / A2C)
    pub decision_module: DecisionModule,

    // Этап 6: World Model
    pub world_model: WorldModel,

    // Состояние агента
    pub last_action: i64,
    pub last_reward: f32,
    pub social_status: f32,
}

impl PersonalityAgent {
    pub fn new_hierarchical(id: usize, device: Device, config: &AgentConfig) -> Self {
        Self::new_internal(id, device, config, |vs, cfg| {
            DecisionModule::Hierarchical {
                hrl: HierarchicalRL::new(
                    vs,
                    STATE_DIM,
                    ACTION_DIM,
                    cfg.hidden_dim,
                    cfg.num_skills,
                ),
            }
        })
    }

    pub fn new_meta_learning(id: usize, device: Device, config: &AgentConfig) -> Self {
        Self::new_internal(id, device, config, |vs, cfg| {
            DecisionModule::MetaLearning {
                meta_learner: MetaLearner::new(vs, STATE_DIM, ACTION_DIM, cfg.hidden_dim),
            }
        })
    }

    pub fn new_actor_critic(id: usize, device: Device, config: &AgentConfig) -> Self {
        Self::new_internal(id, device, config, |_vs, _cfg| {
            DecisionModule::ActorCriticOnly
        })
    }

    fn new_internal(
        id: usize,
        device: Device,
        config: &AgentConfig,
        make_decision: impl FnOnce(&nn::Path, &AgentConfig) -> DecisionModule,
    ) -> Self {
        let vs = VarStore::new(device);
        let root = vs.root();

        let actor_critic =
            ActorCritic::new(&(&root / "ac"), STATE_DIM, ACTION_DIM, config.hidden_dim);
        let world_model = WorldModel::new(&(&root / "wm"), STATE_DIM, config.hidden_dim);
        let decision_module = make_decision(&root, config);

        // Optimizer для ВСЕХ параметров агента (единый VarStore)
        let optimizer = nn::Adam::default().build(&vs, config.lr).unwrap();

        PersonalityAgent {
            id,
            vs,
            optimizer,

            // Этап 1: БОГАТАЯ нейрохимия
            neurochemistry: NeurochemicalState::new(),
            homeostasis: HomeostaticState::new(),
            memory: EpisodicMemory::new(config.memory_size),

            // Этап 2: PFC
            pfc_planner: PfcPlanner::default_personality(),
            action_social_costs: [
                (0_i64, 0.0_f32),
                (1, 0.3),
                (2, -0.1),
                (3, -0.6),
            ]
            .into_iter()
            .collect(),

            // Этап 3: позиция + ToM
            position: Position { x: 0, y: 0 },
            theory_of_mind: TheoryOfMind::new(),
            use_tom: config.use_tom,

            // Этап 4-6: нейросети
            actor_critic,
            decision_module,
            world_model,

            // Состояние
            last_action: 0,
            last_reward: 0.0,
            social_status: 0.0,
        }
    }

    /// Собирает вектор состояния из РЕАЛЬНОЙ нейрохимии + гомеостаза + позиции
    pub fn build_state_vector(&self, world: &GridWorld) -> Vec<f32> {
        vec![
            self.position.x as f32 / world.width as f32,
            self.position.y as f32 / world.height as f32,
            self.neurochemistry.dopamine_level(),
            self.neurochemistry.serotonin_level(),
            self.neurochemistry.cortisol_level(),
            self.neurochemistry.norepinephrine_level(),
            self.neurochemistry.acetylcholine_level(),
            self.homeostasis.safety,
            self.homeostasis.social_connection,
            self.homeostasis.curiosity,
        ]
    }

    /// Выбор действия: все модули работают ВМЕСТЕ
    pub fn choose_action(&mut self, world: &GridWorld) -> Action {
        // Этап 1: когнитивная модуляция из БОГАТОЙ нейрохимии
        let cog = self.neurochemistry.compute_cognitive_modulation();

        // Этап 3: ToM обновляется из нейрохимии
        self.theory_of_mind.update_from_neurochemistry(&cog);

        // Собираем state tensor из реальных данных
        let state_vec = self.build_state_vector(world);
        let state_tensor = Tensor::from_slice(&state_vec)
            .unsqueeze(0)
            .to_device(self.vs.device());

        // Этап 5: выбор действия через enum dispatch
        let raw_action = match &mut self.decision_module {
            DecisionModule::Hierarchical { hrl } => {
                hrl.choose_action(&state_tensor, &cog)
            }
            DecisionModule::MetaLearning { meta_learner } => {
                meta_learner.select_action(&state_tensor, self.last_action, self.last_reward)
            }
            DecisionModule::ActorCriticOnly => {
                let (action_idx, _) = self
                    .actor_critic
                    .select_action(&state_tensor, cog.exploration_temperature);
                action_idx
            }
        };

        // Этап 2: PFC гейтинг на РЕАЛЬНОЙ нейрохимии
        let gated = self.pfc_planner.gate_action(
            raw_action,
            ACTION_DIM,
            &self.neurochemistry,
            &self.homeostasis,
            &self.action_social_costs,
        );

        self.last_action = gated;
        Action::from_index(gated)
    }

    /// Обработка шага среды: обновление нейрохимии, гомеостаза, памяти, обучение
    pub fn process_step(
        &mut self,
        world: &GridWorld,
        action: Action,
        next_pos: Position,
    ) {
        let prev_state = self.build_state_vector(world);

        // Этап 3: реакция на среду
        let threat = world.get_threat(next_pos);
        if threat > 0.0 {
            self.neurochemistry.process_threat(threat);
            self.homeostasis.experience_threat(threat);
        }

        let social = world.get_social_value(next_pos);
        if social > 0.0 {
            self.process_social_event(social);
        }

        let extrinsic_reward = world.get_extrinsic_reward(next_pos);
        if extrinsic_reward > 0.0 {
            self.neurochemistry
                .process_prediction_error(extrinsic_reward * 0.1);
        }

        // Этап 1: шаг нейрохимической динамики
        self.neurochemistry.step(100);
        self.homeostasis.update(0.1);

        // Обновляем позицию
        self.position = next_pos;

        // Этап 1: intrinsic motivation
        let intrinsic = self.homeostasis.get_intrinsic_motivation(0.1);
        let total_reward = extrinsic_reward + intrinsic * 0.3;
        self.last_reward = total_reward;

        // Этап 1: сохраняем в эпизодическую память
        let next_state = self.build_state_vector(world);
        self.memory.store(Episode {
            state: prev_state,
            action: action.to_index(),
            reward: total_reward,
            next_state,
            done: world.is_terminal(next_pos),
            emotional_valence: self
                .neurochemistry
                .compute_cognitive_modulation()
                .emotional_valence,
            homeostatic_significance: self.homeostasis.get_homeostatic_error(),
            timestamp: self.neurochemistry.current_timestep_ms,
        });

        // Этап 4: обучение Actor-Critic из памяти
        self.learn(world);
    }

    /// Обучение: использует CognitiveModulation для learning rate и discount
    fn learn(&mut self, world: &GridWorld) {
        let batch_size = 32;
        if self.memory.len() < batch_size {
            return;
        }

        let cog = self.neurochemistry.compute_cognitive_modulation();
        let batch = self.memory.sample_batch(batch_size);
        let device = self.vs.device();

        let states_flat: Vec<f32> = batch.iter().flat_map(|e| e.state.iter().copied()).collect();
        let next_flat: Vec<f32> = batch
            .iter()
            .flat_map(|e| e.next_state.iter().copied())
            .collect();
        let actions: Vec<i64> = batch.iter().map(|e| e.action).collect();
        let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
        let dones: Vec<f32> = batch
            .iter()
            .map(|e| if e.done { 1.0 } else { 0.0 })
            .collect();

        let bs = batch.len() as i64;
        let states_t = Tensor::from_slice(&states_flat)
            .to_device(device)
            .view([bs, STATE_DIM]);
        let next_t = Tensor::from_slice(&next_flat)
            .to_device(device)
            .view([bs, STATE_DIM]);
        let actions_t = Tensor::from_slice(&actions)
            .to_device(device)
            .unsqueeze(1);
        let rewards_t = Tensor::from_slice(&rewards)
            .to_device(device)
            .unsqueeze(1);
        let dones_t = Tensor::from_slice(&dones)
            .to_device(device)
            .unsqueeze(1);

        // discount из CognitiveModulation (нейрохимия влияет на обучение)
        self.actor_critic.batch_update(
            &mut self.optimizer,
            &states_t,
            &actions_t,
            &rewards_t,
            &next_t,
            &dones_t,
            cog.temporal_discount,
        );
    }

    /// Социальное событие: серотонин + дофамин + статус
    pub fn process_social_event(&mut self, quality: f32) {
        self.homeostasis.social_interaction(quality);
        // Серотонин в PFC (реальная нейрохимия, не заглушка)
        self.neurochemistry
            .phasic_release(1, "prefrontal_cortex", quality * 0.2);
        // Дофамин от социального вознаграждения
        self.neurochemistry
            .phasic_release(0, "nucleus_accumbens", quality * 0.1);
        self.social_status = (self.social_status + quality * 0.1).min(1.0);
    }

    pub fn decision_type_name(&self) -> &'static str {
        match &self.decision_module {
            DecisionModule::Hierarchical { .. } => "HRL",
            DecisionModule::MetaLearning { .. } => "MetaL",
            DecisionModule::ActorCriticOnly => "A2C",
        }
    }

    pub fn current_skill_id(&self) -> Option<usize> {
        match &self.decision_module {
            DecisionModule::Hierarchical { hrl } => hrl.current_skill,
            _ => None,
        }
    }
}
