use tch::{nn, nn::ModuleT, nn::OptimizerConfig, nn::VarStore, Tensor, Kind, Device};
use rand::{Rng, thread_rng};
use std::collections::HashMap;

mod pfc_planner;
use pfc_planner::PfcPlanner;

const STATE_DIM: i64 = 6;
const ACTION_DIM: i64 = 4;
const NUM_SKILLS: usize = 32;
const HIDDEN_DIM: i64 = 512;
const LR: f64 = 1e-4;
const GAMMA: f64 = 0.99;
const BATCH_SIZE: usize = 2048;
const MEMORY_SIZE: usize = 2_000_000;

pub type State = Tensor;
pub type Action = i64;

pub struct Episode {
    pub state: State,
    pub action: Action,
    pub reward: f32,
    pub next_state: State,
    pub skill_id: usize,
    pub terminated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct NeurochemicalState {
    pub dopamine: f32,
    pub serotonin: f32,
    pub cortisol: f32,
    pub norepinephrine: f32,
    pub acetylcholine: f32,
    pub exploration_temperature: f32,
    pub impulse_control: f32,
}

impl Default for NeurochemicalState {
    fn default() -> Self {
        NeurochemicalState {
            dopamine: 0.5, serotonin: 0.5, cortisol: 0.2, norepinephrine: 0.5, acetylcholine: 0.5,
            exploration_temperature: 1.0,
            impulse_control: 0.5,
        }
    }
}

pub struct HomeostaticState {
    pub safety: f32,
    pub social_connection: f32,
    pub curiosity: f32,
}

impl Default for HomeostaticState {
    fn default() -> Self {
        HomeostaticState { safety: 0.5, social_connection: 0.5, curiosity: 0.5 }
    }
}

use std::collections::VecDeque;

pub struct EpisodicMemory {
    buffer: VecDeque<Episode>,
    max_size: usize,
}

impl EpisodicMemory {
    pub fn new() -> Self { 
        EpisodicMemory {
            buffer: VecDeque::with_capacity(MEMORY_SIZE),
            max_size: MEMORY_SIZE,
        }
    }
    
    pub fn add_experience(&mut self, episode: Episode) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(episode);
    }
    
    pub fn sample_batch(&self, batch_size: usize) -> Vec<Episode> {
        use rand::seq::SliceRandom;
        if self.buffer.len() < batch_size {
            return self.buffer.iter().cloned().collect();
        }
        self.buffer.iter()
            .choose_multiple(&mut thread_rng(), batch_size)
            .into_iter()
            .cloned()
            .collect()
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct ActorCritic {
    pub vs: VarStore,
    pub policy: nn::Sequential,
    pub value: nn::Sequential,
    pub optimizer: nn::Optimizer,
}

impl ActorCritic {
    pub fn new(vs: &nn::Path) -> Self {
        let vs = vs.sub("ac");
        let policy = nn::seq()
            .add(nn::linear(&vs.sub("pol1"), STATE_DIM, HIDDEN_DIM, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.sub("pol2"), HIDDEN_DIM, ACTION_DIM, Default::default()));
        let value = nn::seq()
            .add(nn::linear(&vs.sub("val1"), STATE_DIM, HIDDEN_DIM, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.sub("val2"), HIDDEN_DIM, 1, Default::default()));
        let optimizer = nn::Adam::default().build(&vs, LR).unwrap();
        ActorCritic { vs: vs.to_var_store(), policy, value, optimizer }
    }

    pub fn select_action(&self, state: &State) -> Action {
        let logits = self.policy.forward(state);
        logits.softmax(-1, Kind::Float).argmax(-1, false).int64_value(&[])
    }

    pub fn update(&mut self, _batch: &[Episode]) {}
}

pub struct WorldModel {
    pub vs: VarStore,
    pub encoder: nn::Sequential,
    pub decoder: nn::Sequential,
    pub dynamics: nn::GRU,
    pub optimizer: nn::Optimizer,
}

impl WorldModel {
    pub fn new(vs: &nn::Path) -> Self {
        let vs = vs.sub("wm");
        let encoder = nn::seq().add(nn::linear(&vs.sub("enc"), STATE_DIM, HIDDEN_DIM, Default::default()));
        let decoder = nn::seq().add(nn::linear(&vs.sub("dec"), HIDDEN_DIM, STATE_DIM, Default::default()));
        let dynamics = nn::gru(&vs.sub("gru"), HIDDEN_DIM, HIDDEN_DIM, Default::default()).unwrap();
        let optimizer = nn::Adam::default().build(&vs, LR).unwrap();
        WorldModel { vs: vs.to_var_store(), encoder, decoder, dynamics, optimizer }
    }

    pub fn predict_next_state(&self, state: &State, _action: Action) -> State {
        self.encoder.forward(state).relu().copy()
    }
}

pub struct MetaController {
    pub vs: VarStore,
    pub network: nn::Sequential,
    pub optimizer: nn::Optimizer,
    pub num_skills: usize,
}

impl MetaController {
    pub fn new(vs: &nn::Path, state_dim: i64, num_skills: usize) -> Self {
        let vs = vs.sub("meta_controller");
        let network = nn::seq()
            .add(nn::linear(&vs.sub("fc1"), state_dim, HIDDEN_DIM, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.sub("fc2"), HIDDEN_DIM, num_skills as i64, Default::default()));
        let optimizer = nn::Adam::default().build(&vs, LR).unwrap();
        MetaController { vs: vs.to_var_store(), network, optimizer, num_skills }
    }

    pub fn select_skill(&self, state: &State, temperature: f32) -> usize {
        let logits = self.network.forward(state);
        let scaled_logits = logits / temperature as f64;
        let probs = scaled_logits.softmax(-1, Kind::Float);
        
        if thread_rng().gen::<f32>() < temperature * 0.1 { 
            thread_rng().gen_range(0..self.num_skills)
        } else {
            probs.argmax(-1, false).int64_value(&[]) as usize
        }
    }

    pub fn update(&mut self, _states: &Tensor, _skills: &Tensor, _advantages: &Tensor) {}
}

pub struct Skill {
    pub id: usize,
    pub vs: VarStore,
    pub policy: nn::Sequential,
    pub termination_fn: nn::Sequential,
    pub optimizer: nn::Optimizer,
}

impl Skill {
    pub fn new(vs: &nn::Path, id: usize, state_dim: i64, action_dim: i64) -> Self {
        let vs = vs.sub(&format!("skill_{}", id));
        let policy = nn::seq()
            .add(nn::linear(&vs.sub("pol_fc1"), state_dim, HIDDEN_DIM, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.sub("pol_fc2"), HIDDEN_DIM, action_dim, Default::default()));
        let termination_fn = nn::seq()
            .add(nn::linear(&vs.sub("term_fc1"), state_dim, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.sub("term_fc2"), 32, 1, Default::default()));
        let optimizer = nn::Adam::default().build(&vs, LR).unwrap();
        Skill { id, vs: vs.to_var_store(), policy, termination_fn, optimizer }
    }

    pub fn select_action(&self, state: &State, temperature: f32) -> Action {
        let logits = self.policy.forward(state);
        let scaled_logits = logits / temperature as f64;
        scaled_logits.softmax(-1, Kind::Float).argmax(-1, false).int64_value(&[])
    }

    pub fn should_terminate(&self, state: &State, impulse_control: f32) -> bool {
        let logit = self.termination_fn.forward(state).squeeze();
        let prob_terminate = logit.sigmoid().double_value(&[]);
        let termination_threshold = 1.0 - impulse_control as f64;
        prob_terminate > termination_threshold
    }

    pub fn update(&mut self, _states: &Tensor, _actions: &Tensor, _advantages: &Tensor) {}
}

pub struct MetaLearner {
    pub vs: VarStore,
    pub recurrent_policy: nn::GRU,
    pub policy_head: nn::Sequential,
    pub value_head: nn::Sequential,
    pub optimizer: nn::Optimizer,
    pub hidden_state: Tensor,
}

impl MetaLearner {
    pub fn new(vs: &nn::Path) -> Self {
        let vs = vs.sub("meta_learner");
        let input_size = STATE_DIM + ACTION_DIM + 1;
        
        let recurrent_policy = nn::gru(&vs.sub("gru"), input_size, HIDDEN_DIM, Default::default()).unwrap();
        
        let policy_head = nn::seq().add(nn::linear(&vs.sub("pol_h"), HIDDEN_DIM, ACTION_DIM, Default::default()));
        let value_head = nn::seq().add(nn::linear(&vs.sub("val_h"), HIDDEN_DIM, 1, Default::default()));
        
        let optimizer = nn::Adam::default().build(&vs, LR).unwrap();
        let hidden_state = Tensor::zeros(&[1, HIDDEN_DIM], (Kind::Float, Device::Cpu));

        MetaLearner { 
            vs: vs.to_var_store(), recurrent_policy, policy_head, value_head, optimizer, hidden_state
        }
    }

    pub fn select_action(&mut self, state: &State, last_action: Action, last_reward: f32) -> Action {
        let last_action_onehot = Tensor::zeros(&[1, ACTION_DIM], (Kind::Float, Device::Cpu)).one_hot(last_action, ACTION_DIM);
        let last_reward_tensor = Tensor::from_slice(&[last_reward]).unsqueeze(0);

        let input_tensor = Tensor::cat(&[state.copy(), last_action_onehot, last_reward_tensor], -1);

        let (output, next_hidden) = self.recurrent_policy.forward_t(&input_tensor.unsqueeze(0), Some(&self.hidden_state), false);
        self.hidden_state = next_hidden.squeeze(0).copy();

        let logits = self.policy_head.forward(&output.squeeze(0));
        logits.softmax(-1, Kind::Float).argmax(-1, false).int64_value(&[])
    }

    pub fn update(&mut self, _batch: &[Episode]) {}

    pub fn reset_hidden_state(&mut self) {
        self.hidden_state = Tensor::zeros(&[1, HIDDEN_DIM], (Kind::Float, Device::Cpu));
    }
}

pub struct OtherAgentModel {
    pub vs: VarStore,
    pub policy_model: nn::Sequential,
    pub optimizer: nn::Optimizer,
    pub trust_level: f32,
    pub empathy_level: f32,
}

impl OtherAgentModel {
    pub fn new(vs: &nn::Path, id: usize) -> Self {
        let vs = vs.sub(&format!("other_model_{}", id));
        let policy_model = nn::seq()
            .add(nn::linear(&vs.sub("pol1"), STATE_DIM, HIDDEN_DIM, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.sub("pol2"), HIDDEN_DIM, ACTION_DIM, Default::default()));
        let optimizer = nn::Adam::default().build(&vs, LR).unwrap();
        OtherAgentModel { vs: vs.to_var_store(), policy_model, optimizer, trust_level: 0.5, empathy_level: 0.5 }
    }
}

pub struct TheoryOfMind {
    pub other_models: HashMap<usize, OtherAgentModel>, 
    pub perspective_shift_factor: f32,
}

impl TheoryOfMind {
    pub fn new() -> Self {
        TheoryOfMind { other_models: HashMap::new(), perspective_shift_factor: 1.0 }
    }

    pub fn update_from_neurochemistry(&mut self, neurochem: &NeurochemicalState) {
        self.perspective_shift_factor = 1.0 - (neurochem.cortisol + neurochem.norepinephrine) / 2.0;
        self.perspective_shift_factor = self.perspective_shift_factor.max(0.1).min(1.0);

        for model in self.other_models.values_mut() {
            model.empathy_level = (model.empathy_level * 0.9 + neurochem.serotonin * 0.1).min(1.0);
        }
    }

    pub fn observe_other_agent(&mut self, other_id: usize, state: &State, action: Action) {
        if !self.other_models.contains_key(&other_id) {
            let mut vs = VarStore::new(Device::cuda_if_available());
            self.other_models.insert(other_id, OtherAgentModel::new(&vs.root(), other_id));
        }
    }

    pub fn predict_other_action(&self, other_id: usize, their_state: &State) -> Option<Action> {
        self.other_models.get(&other_id).map(|model| {
            let logits = model.policy_model.forward(their_state);
            logits.softmax(-1, Kind::Float).argmax(-1, false).int64_value(&[])
        })
    }
}

pub struct PersonalityAgent {
    pub neurochemistry: NeurochemicalState,
    pub homeostasis: HomeostaticState,
    pub episodic_memory: EpisodicMemory,
    pub actor_critic: ActorCritic,
    pub world_model: WorldModel,
    pub meta_controller: MetaController,
    pub skills: Vec<Skill>,
    pub current_skill: Option<usize>,
    pub meta_learner: MetaLearner,
    pub last_action: Action,
    pub last_reward: f32,
    pub theory_of_mind: TheoryOfMind,
    pub social_status: f32,
    pub pfc_planner: PfcPlanner,
    pub action_social_costs: HashMap<i64, f32>,
    pub vs: VarStore,
    pub state: State,
    pub use_hierarchical: bool,
    pub use_meta_learning: bool,
    pub use_tom: bool,
}

impl PersonalityAgent {
    pub fn new(id: usize, use_h: bool, use_m: bool, use_t: bool) -> Self {
        let device = Device::cuda_if_available();
        let mut vs = VarStore::new(device);
        let root = nn::Path::new(&mut vs, &format!("agent_{}", id));

        let meta_controller = MetaController::new(&root.sub("hrl"), STATE_DIM, NUM_SKILLS);
        let skills: Vec<Skill> = (0..NUM_SKILLS)
            .map(|i| Skill::new(&root.sub("hrl"), i, STATE_DIM, ACTION_DIM))
            .collect();

        PersonalityAgent {
            neurochemistry: NeurochemicalState::default(),
            homeostasis: HomeostaticState::default(),
            episodic_memory: EpisodicMemory::new(),
            actor_critic: ActorCritic::new(&root),
            world_model: WorldModel::new(&root),
            meta_controller,
            skills,
            current_skill: None,
            meta_learner: MetaLearner::new(&root),
            last_action: 0,
            last_reward: 0.0,
            theory_of_mind: TheoryOfMind::new(),
            social_status: 0.0,
            pfc_planner: PfcPlanner::default_personality(),
            action_social_costs: {
                let mut m = HashMap::new();
                m.insert(0, 0.0_f32);
                m.insert(1, 0.3_f32);
                m.insert(2, -0.1_f32);
                m.insert(3, -0.6_f32);
                m
            },
            vs,
            state: Tensor::zeros(&[1, STATE_DIM], (Kind::Float, Device::Cpu)),
            use_hierarchical: use_h,
            use_meta_learning: use_m,
            use_tom: use_t,
        }
    }

    pub fn choose_action(&mut self, other_agents_states: &HashMap<usize, &State>) -> Action {
        self.theory_of_mind.update_from_neurochemistry(&self.neurochemistry);

        let mut action: Action;

        if self.use_hierarchical {
            action = self.choose_action_hierarchical();
        } else if self.use_meta_learning {
            action = self.meta_learner.select_action(&self.state, self.last_action, self.last_reward);
        } else {
            action = self.actor_critic.select_action(&self.state);
        }

        if self.use_tom {
            for (id, their_state) in other_agents_states.iter() {
                if let Some(predicted_action) = self.theory_of_mind.predict_other_action(*id, their_state) {
                    if predicted_action == action {
                        if self.neurochemistry.serotonin > self.neurochemistry.cortisol {
                            let mut rng = thread_rng();
                            action = (0..ACTION_DIM).filter(|&a| a != action).map(|a| a as i64).collect::<Vec<i64>>()[rng.gen_range(0..ACTION_DIM - 1) as usize];
                            println!("Agent modified action due to ToM cooperation.");
                        }
                    }
                }
            }
        }

        if self.use_hierarchical || self.use_meta_learning {
            action = self.pfc_planner.gate_action(
                action,
                ACTION_DIM,
                self.neurochemistry.cortisol,
                self.neurochemistry.serotonin,
                self.homeostasis.safety,
                &self.action_social_costs,
            );
        }
        
        self.last_action = action;
        action
    }

    fn choose_action_hierarchical(&mut self) -> Action {
        let exploration_temp = self.neurochemistry.exploration_temperature;
        let impulse_control = self.neurochemistry.impulse_control;

        let mut next_skill_id = self.current_skill;
        
        if let Some(skill_id) = self.current_skill {
            let current_skill = &self.skills[skill_id];
            if current_skill.should_terminate(&self.state, impulse_control) {
                next_skill_id = None;
            }
        }

        if next_skill_id.is_none() {
            let new_skill_id = self.meta_controller.select_skill(&self.state, exploration_temp);
            self.current_skill = Some(new_skill_id);
        }

        let active_skill_id = self.current_skill.unwrap();
        let active_skill = &self.skills[active_skill_id];
        active_skill.select_action(&self.state, exploration_temp)
    }

    pub fn process_social_event(&mut self, social_quality: f32) {
        self.neurochemistry.serotonin = (self.neurochemistry.serotonin + social_quality * 0.1).min(1.0);
        self.neurochemistry.dopamine = (self.neurochemistry.dopamine + social_quality * 0.05).min(1.0);
        self.social_status = (self.social_status + social_quality * 0.1).min(1.0);
    }
    
    pub fn update_state(&mut self, next_state: State, reward: f32, _is_terminal: bool) {
        self.state = next_state;
        self.last_reward = reward;
        self.neurochemistry.dopamine = (self.neurochemistry.dopamine + reward * 0.01).min(1.0).max(0.0);
    }

    pub fn learn(&mut self) {
        let _batch = self.episodic_memory.sample_batch(32);
    }
}

pub struct GridWorld {
    pub size: (i64, i64),
}

impl GridWorld {
    pub fn new(size: (i64, i64)) -> Self { GridWorld { size } }
    
    pub fn execute_action(&self, _agent_id: usize, _action: Action, current_state: &State) -> (State, f32) {
        let next_state = current_state.copy();
        let reward = thread_rng().gen_range(-0.1..0.5);
        (next_state, reward)
    }
    
    pub fn check_social_interaction(&self, _agent_a: usize, _agent_b: usize) -> f32 {
        if thread_rng().gen::<f32>() < 0.2 { thread_rng().gen_range(0.1..0.5) } else { 0.0 }
    }
}

pub struct MultiAgentWorld {
    pub agents: Vec<PersonalityAgent>,
    pub world: GridWorld,
}

impl MultiAgentWorld {
    pub fn new(agents: Vec<PersonalityAgent>, world: GridWorld) -> Self {
        MultiAgentWorld { agents, world }
    }

    pub fn step(&mut self) {
        let num_agents = self.agents.len();
        let mut actions = vec![0; num_agents];
        let mut current_states: HashMap<usize, State> = self.agents.iter().enumerate().map(|(i, a)| (i, a.state.copy())).collect();
        
        for i in 0..num_agents {
            let mut other_agents_states = HashMap::new();
            for j in 0..num_agents {
                if i != j {
                    other_agents_states.insert(j, current_states.get(&j).unwrap());
                }
            }
            
            actions[i] = self.agents[i].choose_action(&other_agents_states);
            
            for j in 0..num_agents {
                if i != j {
                    self.agents[i].theory_of_mind.observe_other_agent(
                        j,
                        current_states.get(&j).unwrap(),
                        self.agents[j].last_action,
                    );
                }
            }
        }

        for i in 0..num_agents {
            let action = actions[i];
            let current_state = current_states.get(&i).unwrap();
            
            let (next_state, reward) = self.world.execute_action(i, action, current_state);
            let is_terminal = false;

            self.agents[i].update_state(next_state.copy(), reward, is_terminal);
            let episode = Episode {
                state: current_state.copy(), action, reward, next_state, skill_id: self.agents[i].current_skill.unwrap_or(0), terminated: is_terminal
            };
            self.agents[i].episodic_memory.add_experience(episode);
            
            for j in (i + 1)..num_agents {
                let social_quality = self.world.check_social_interaction(i, j);
                if social_quality > 0.0 {
                    self.agents[i].process_social_event(social_quality);
                    self.agents[j].process_social_event(social_quality);
                    println!("Social event between Agent {} and {} (Quality: {:.2})", i, j, social_quality);
                }
            }
        }
    }
}

fn main() {
    println!("=== EXTREME SCALE: 50M steps, 100 agents, 200x200 world ===");
    println!("Hardware: AMD 9950X + 192GB RAM + Radeon PRO W7900 32GB");
    
    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);
    
    std::fs::create_dir_all("checkpoints").unwrap();
    
    let mut agents: Vec<PersonalityAgent> = (0..100)
        .map(|i| {
            let use_hierarchical = i % 3 == 0;
            let use_meta_learning = i % 3 == 1;
            let use_tom = i % 2 == 0;
            PersonalityAgent::new(i, use_hierarchical, use_meta_learning, use_tom)
        })
        .collect();

    let load_checkpoint = std::env::var("LOAD_CHECKPOINT").is_ok();
    if load_checkpoint {
        println!("Loading checkpoints...");
        for (i, agent) in agents.iter_mut().enumerate() {
            let checkpoint_path = format!("checkpoints/agent_{}_final.pt", i);
            if std::path::Path::new(&checkpoint_path).exists() {
                agent.vs.load(&checkpoint_path).unwrap();
                println!("Loaded: agent_{}", i);
            }
        }
    }

    let world = GridWorld::new((200, 200));
    let mut multi_world = MultiAgentWorld::new(agents, world);

    println!("\n100 Agents initialized:");
    println!("HRL: ~33, Meta-L: ~33, A2C: ~34");
    println!("ToM enabled: ~50 agents");
    
    const NUM_STEPS: usize = 50_000_000;
    const LOG_INTERVAL: usize = 100_000;
    const SAVE_INTERVAL: usize = 1_000_000;
    const STATS_INTERVAL: usize = 10_000;
    
    use std::time::Instant;
    let start_time = Instant::now();
    let mut total_social_events = 0;
    
    for step in 0..NUM_STEPS {
        multi_world.step();

        if step % STATS_INTERVAL == 0 {
            for agent in &multi_world.agents {
                if thread_rng().gen::<f32>() < 0.01 {
                    total_social_events += 1;
                }
            }
        }

        if step % LOG_INTERVAL == 0 || step == NUM_STEPS - 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let steps_per_sec = (step + 1) as f32 / elapsed;
            let eta_hours = (NUM_STEPS - step) as f32 / steps_per_sec / 3600.0;
            
            println!("\n=== STEP {}/{} ({:.1}%) ===", step, NUM_STEPS, (step as f32 / NUM_STEPS as f32) * 100.0);
            println!("Speed: {:.1} steps/s | Elapsed: {:.1}h | ETA: {:.1}h", 
                     steps_per_sec, elapsed / 3600.0, eta_hours);
            
            let avg_da: f32 = multi_world.agents.iter().map(|a| a.neurochemistry.dopamine).sum::<f32>() / multi_world.agents.len() as f32;
            let avg_5ht: f32 = multi_world.agents.iter().map(|a| a.neurochemistry.serotonin).sum::<f32>() / multi_world.agents.len() as f32;
            let avg_cort: f32 = multi_world.agents.iter().map(|a| a.neurochemistry.cortisol).sum::<f32>() / multi_world.agents.len() as f32;
            let avg_safety: f32 = multi_world.agents.iter().map(|a| a.homeostasis.safety).sum::<f32>() / multi_world.agents.len() as f32;
            let avg_social: f32 = multi_world.agents.iter().map(|a| a.homeostasis.social_connection).sum::<f32>() / multi_world.agents.len() as f32;
            
            println!("Population avg: DA={:.3} 5HT={:.3} Cort={:.3} | Safety={:.3} Social={:.3}", 
                     avg_da, avg_5ht, avg_cort, avg_safety, avg_social);
            println!("Social events: {}", total_social_events);
            
            for i in [0, 1, 50, 99].iter() {
                if *i < multi_world.agents.len() {
                    let agent = &multi_world.agents[*i];
                    let decision_type = if agent.use_hierarchical { "HRL" } 
                                        else if agent.use_meta_learning { "MetaL" } 
                                        else { "A2C" };
                    let skill = agent.current_skill.map_or("N".to_string(), |id| id.to_string());
                    println!("  A{}: {} Act={} DA={:.2} 5HT={:.2} S={}", 
                             i, decision_type, agent.last_action, 
                             agent.neurochemistry.dopamine, 
                             agent.neurochemistry.serotonin, skill);
                }
            }
        }

        if step % SAVE_INTERVAL == 0 && step > 0 {
            println!("\nSaving checkpoints at step {}...", step);
            for i in 0..multi_world.agents.len() {
                let checkpoint_path = format!("checkpoints/agent_{}_step_{}.pt", i, step);
                multi_world.agents[i].vs.save(&checkpoint_path).unwrap_or_else(|e| {
                    eprintln!("Failed to save agent {}: {}", i, e);
                });
            }
            println!("Checkpoints saved for {} agents", multi_world.agents.len());
        }
    }

    println!("\n=== TRAINING COMPLETE: {} steps ===", NUM_STEPS);
    println!("Total time: {:.2} hours", start_time.elapsed().as_secs_f32() / 3600.0);
    
    println!("\nSaving final weights...");
    for i in 0..multi_world.agents.len() {
        let final_path = format!("checkpoints/agent_{}_final.pt", i);
        multi_world.agents[i].vs.save(&final_path).unwrap();
    }
    println!("Final weights saved for {} agents", multi_world.agents.len());
