use std::collections::{HashMap, VecDeque};
use rand::Rng;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

#[derive(Clone, Debug)]
struct ReceptorState {
    sensitivity: f32,
    desensitization_rate: f32,
    current_desensitization: f32,
    binding_affinity: f32,
}

#[derive(Clone, Debug)]
struct NeurotransmitterRegion {
    concentration: f32,
    baseline: f32,
    release_rate: f32,
    reuptake_rate: f32,
    decay_halflife_ms: f32,
    receptors: HashMap<String, ReceptorState>,
}

#[derive(Clone, Debug)]
struct NeuromodulatorSystem {
    regions: HashMap<String, NeurotransmitterRegion>,
    temporal_scale: TemporalScale,
}

#[derive(Clone, Debug)]
enum TemporalScale {
    Fast,
    Medium,
    Slow,
}

#[derive(Clone, Debug)]
struct InteractionRule {
    source_system: usize,
    source_region: String,
    target_system: usize,
    target_region: String,
    modulation_type: ModulationType,
    strength: f32,
    threshold: f32,
    time_constant_ms: f32,
}

#[derive(Clone, Debug)]
enum ModulationType {
    Facilitation,
    Inhibition,
    SensitivityModulation,
    ReleaseModulation,
    ReceptorDensityChange,
}

struct NeurochemicalState {
    dopamine: NeuromodulatorSystem,
    serotonin: NeuromodulatorSystem,
    norepinephrine: NeuromodulatorSystem,
    cortisol: NeuromodulatorSystem,
    acetylcholine: NeuromodulatorSystem,
    interaction_rules: Vec<InteractionRule>,
    current_timestep_ms: u64,
}

#[derive(Clone, Debug)]
struct CognitiveModulation {
    learning_rate_multiplier: f32,
    exploration_temperature: f32,
    risk_sensitivity: f32,
    working_memory_capacity: f32,
    emotional_valence: f32,
    temporal_discount: f32,
    attention_focus: f32,
    impulse_control: f32,
}

#[derive(Clone, Debug)]
struct HomeostaticState {
    safety: f32,
    safety_setpoint: f32,
    social_connection: f32,
    social_setpoint: f32,
    curiosity: f32,
    curiosity_decay: f32,
}

impl HomeostaticState {
    fn new() -> Self {
        HomeostaticState {
            safety: 1.0,
            safety_setpoint: 0.8,
            social_connection: 0.5,
            social_setpoint: 0.6,
            curiosity: 1.0,
            curiosity_decay: 0.998,
        }
    }

    fn update(&mut self, dt: f32) {
        self.safety = (self.safety + 0.01 * dt).min(1.0);
        self.social_connection -= 0.005 * dt;
        self.social_connection = self.social_connection.max(0.0).min(1.0);
        self.curiosity *= self.curiosity_decay.powf(dt);
    }

    fn get_homeostatic_error(&self) -> f32 {
        let safety_error = (self.safety_setpoint - self.safety).max(0.0);
        let social_error = (self.social_setpoint - self.social_connection).max(0.0);
        safety_error * 1.5 + social_error * 0.8
    }

    fn get_intrinsic_motivation(&self, state_novelty: f32) -> f32 {
        let homeostatic_drive = self.get_homeostatic_error();
        let curiosity_drive = self.curiosity * state_novelty;
        homeostatic_drive + curiosity_drive * 0.5
    }

    fn experience_threat(&mut self, intensity: f32) {
        self.safety -= intensity;
        self.safety = self.safety.max(0.0);
    }

    fn social_interaction(&mut self, quality: f32) {
        self.social_connection = (self.social_connection + quality).min(1.0);
        self.curiosity = (self.curiosity + 0.05).min(1.0);
    }

    fn explore_success(&mut self) {
        self.curiosity = (self.curiosity + 0.1).min(1.0);
    }
}

#[derive(Clone, Debug)]
struct EpisodicMemory {
    buffer: VecDeque<Episode>,
    max_size: usize,
}

#[derive(Clone, Debug)]
struct Episode {
    state: Vec<f32>,
    action: i64,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
    emotional_valence: f32,
    homeostatic_significance: f32,
    timestamp: u64,
}

impl EpisodicMemory {
    fn new(max_size: usize) -> Self {
        EpisodicMemory {
            buffer: VecDeque::new(),
            max_size,
        }
    }

    fn store(&mut self, episode: Episode) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(episode);
    }

    fn sample_batch(&self, batch_size: usize) -> Vec<Episode> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::new();
        
        for _ in 0..batch_size.min(self.buffer.len()) {
            let idx = rng.gen_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }
        
        samples
    }
}

#[derive(Clone, Debug)]
struct TheoryOfMind {
    other_agent_models: HashMap<usize, OtherAgentModel>,
    mentalizing_capacity: f32,
}

#[derive(Clone, Debug)]
struct OtherAgentModel {
    current_state: State,
    estimated_goals: Vec<State>,
    estimated_threat_sensitivity: f32,
    estimated_social_orientation: f32,
    confidence: f32,
    interaction_count: usize,
}

impl TheoryOfMind {
    fn new() -> Self {
        TheoryOfMind {
            other_agent_models: HashMap::new(),
            mentalizing_capacity: 1.0,
        }
    }

    fn observe_other_agent(&mut self, agent_id: usize, observed_state: State, observed_action: Action) {
        let model = self.other_agent_models.entry(agent_id).or_insert(OtherAgentModel {
            current_state: observed_state,
            estimated_goals: Vec::new(),
            estimated_threat_sensitivity: 0.5,
            estimated_social_orientation: 0.5,
            confidence: 0.0,
            interaction_count: 0,
        });

        model.interaction_count += 1;
        model.confidence = (model.confidence + 0.1).min(1.0);
        model.current_state = observed_state;

        let inferred_goal = match observed_action {
            Action::Up => State { x: observed_state.x, y: observed_state.y - 1 },
            Action::Down => State { x: observed_state.x, y: observed_state.y + 1 },
            Action::Left => State { x: observed_state.x - 1, y: observed_state.y },
            Action::Right => State { x: observed_state.x + 1, y: observed_state.y },
        };

        if !model.estimated_goals.contains(&inferred_goal) {
            model.estimated_goals.push(inferred_goal);
        }
        
        if model.estimated_goals.len() > 5 {
            model.estimated_goals.remove(0);
        }
    }

    fn predict_other_action(&self, agent_id: usize, current_state: State) -> Option<Action> {
        let model = self.other_agent_models.get(&agent_id)?;
        
        if model.estimated_goals.is_empty() || model.confidence < 0.3 {
            return None;
        }

        let nearest_goal = model.estimated_goals.iter()
            .min_by_key(|goal| {
                let dx = (goal.x - model.current_state.x).abs();
                let dy = (goal.y - model.current_state.y).abs();
                dx + dy
            })?;

        let dx = nearest_goal.x - model.current_state.x;
        let dy = nearest_goal.y - model.current_state.y;

        if dx.abs() > dy.abs() {
            if dx > 0 { Some(Action::Right) } else { Some(Action::Left) }
        } else {
            if dy > 0 { Some(Action::Down) } else { Some(Action::Up) }
        }
    }

    fn should_cooperate(&self, agent_id: usize) -> bool {
        if let Some(model) = self.other_agent_models.get(&agent_id) {
            model.estimated_social_orientation > 0.6 && model.confidence > 0.4
        } else {
            false
        }
    }

    fn update_from_neurochemistry(&mut self, cog_mod: &CognitiveModulation) {
        let serotonin_factor = (1.0 + cog_mod.emotional_valence).max(0.0).min(2.0);
        let base_capacity = 0.5 + (serotonin_factor - 1.0) * 0.3;
        
        let working_memory_factor = (cog_mod.working_memory_capacity / 7.0).min(1.0);
        
        self.mentalizing_capacity = (base_capacity * working_memory_factor).clamp(0.1, 1.0);
    }
}

struct ActorCritic {
    actor: nn::Sequential,
    critic: nn::Sequential,
    actor_optimizer: nn::Optimizer,
    critic_optimizer: nn::Optimizer,
    device: Device,
}

impl ActorCritic {
    fn new(vs: &nn::Path, state_dim: i64, action_dim: i64, learning_rate: f64) -> Self {
        let actor = nn::seq()
            .add(nn::linear(vs / "actor_fc1", state_dim, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "actor_fc2", 128, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "actor_fc3", 64, action_dim, Default::default()));
        
        let critic = nn::seq()
            .add(nn::linear(vs / "critic_fc1", state_dim, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "critic_fc2", 128, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "critic_fc3", 64, 1, Default::default()));
        
        let actor_optimizer = nn::Adam::default().build(vs, learning_rate).unwrap();
        let critic_optimizer = nn::Adam::default().build(vs, learning_rate).unwrap();
        
        ActorCritic {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            device: vs.device(),
        }
    }
    
    fn select_action(&self, state: &[f32], temperature: f32) -> (i64, Tensor) {
        let state_tensor = Tensor::of_slice(state)
            .to_device(self.device)
            .unsqueeze(0);
        
        let logits = self.actor.forward(&state_tensor) / temperature;
        let probs = logits.softmax(-1, tch::Kind::Float);
        
        let action = probs.multinomial(1, true);
        let action_idx = i64::from(action.int64_value(&[0, 0]));
        
        (action_idx, probs)
    }
    
    fn get_value(&self, state: &[f32]) -> f32 {
        let state_tensor = Tensor::of_slice(state)
            .to_device(self.device)
            .unsqueeze(0);
        
        let value = self.critic.forward(&state_tensor);
        f32::from(value.double_value(&[0, 0]))
    }
    
    fn update(&mut self, 
              states: &[Vec<f32>], 
              actions: &[i64], 
              rewards: &[f32],
              next_states: &[Vec<f32>],
              dones: &[bool],
              discount: f32,
              learning_rate_mult: f32) {
        
        let batch_size = states.len() as i64;
        
        let states_flat: Vec<f32> = states.iter().flatten().copied().collect();
        let next_states_flat: Vec<f32> = next_states.iter().flatten().copied().collect();
        let state_dim = states[0].len() as i64;
        
        let states_tensor = Tensor::of_slice(&states_flat)
            .to_device(self.device)
            .view([batch_size, state_dim]);
        
        let next_states_tensor = Tensor::of_slice(&next_states_flat)
            .to_device(self.device)
            .view([batch_size, state_dim]);
        
        let actions_tensor = Tensor::of_slice(actions)
            .to_device(self.device)
            .unsqueeze(1);
        
        let rewards_tensor = Tensor::of_slice(rewards)
            .to_device(self.device)
            .unsqueeze(1);
        
        let dones_tensor = Tensor::of_slice(
            &dones.iter().map(|&d| if d { 1.0f32 } else { 0.0f32 }).collect::<Vec<_>>()
        ).to_device(self.device).unsqueeze(1);
        
        let values = self.critic.forward(&states_tensor);
        let next_values = self.critic.forward(&next_states_tensor).detach();
        
        let targets = &rewards_tensor + discount * &next_values * (1.0 - &dones_tensor);
        let td_errors = &targets - &values;
        
        let critic_loss = td_errors.pow_tensor_scalar(2).mean(tch::Kind::Float);
        
        self.critic_optimizer.zero_grad();
        critic_loss.backward();
        self.critic_optimizer.step();
        
        let logits = self.actor.forward(&states_tensor);
        let log_probs = logits.log_softmax(-1, tch::Kind::Float);
        let action_log_probs = log_probs.gather(1, &actions_tensor, false);
        
        let advantages = td_errors.detach();
        let actor_loss = -(action_log_probs * advantages).mean(tch::Kind::Float);
        
        self.actor_optimizer.zero_grad();
        actor_loss.backward();
        
        self.actor_optimizer.step();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    fn to_index(&self) -> i64 {
        match self {
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
        }
    }
    
    fn from_index(idx: i64) -> Self {
        match idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => Action::Up,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State {
    x: i32,
    y: i32,
}

impl State {
    fn to_vector(&self, world_width: i32, world_height: i32) -> Vec<f32> {
        vec![
            self.x as f32 / world_width as f32,
            self.y as f32 / world_height as f32,
        ]
    }
}

#[derive(Clone, Debug)]
struct GridWorld {
    width: i32,
    height: i32,
    goal: State,
    obstacles: Vec<State>,
    threats: HashMap<State, f32>,
    social_zones: HashMap<State, f32>,
}

struct PersonalityAgent {
    neurochemistry: NeurochemicalState,
    homeostasis: HomeostaticState,
    episodic_memory: EpisodicMemory,
    theory_of_mind: TheoryOfMind,
    actor_critic: ActorCritic,
    state: State,
    base_learning_rate: f64,
    base_exploration: f32,
    episode_count: usize,
    total_reward: f32,
    intrinsic_reward_weight: f32,
    social_status: f32,
    world_width: i32,
    world_height: i32,
}

impl NeurochemicalState {
    fn new() -> Self {
        let dopamine = NeuromodulatorSystem {
            regions: HashMap::from([
                ("nucleus_accumbens".to_string(), NeurotransmitterRegion {
                    concentration: 0.4, baseline: 0.4, release_rate: 0.0,
                    reuptake_rate: 0.85, decay_halflife_ms: 45.0,
                    receptors: HashMap::from([
                        ("D1".to_string(), ReceptorState {
                            sensitivity: 1.1, desensitization_rate: 0.018,
                            current_desensitization: 0.0, binding_affinity: 0.85,
                        }),
                    ]),
                }),
            ]),
            temporal_scale: TemporalScale::Fast,
        };

        let serotonin = NeuromodulatorSystem {
            regions: HashMap::from([
                ("prefrontal_cortex".to_string(), NeurotransmitterRegion {
                    concentration: 0.6, baseline: 0.6, release_rate: 0.0,
                    reuptake_rate: 0.3, decay_halflife_ms: 3_600_000.0,
                    receptors: HashMap::from([
                        ("5HT1A".to_string(), ReceptorState {
                            sensitivity: 1.0, desensitization_rate: 0.001,
                            current_desensitization: 0.0, binding_affinity: 0.8,
                        }),
                    ]),
                }),
            ]),
            temporal_scale: TemporalScale::Medium,
        };

        let norepinephrine = NeuromodulatorSystem {
            regions: HashMap::from([
                ("locus_coeruleus".to_string(), NeurotransmitterRegion {
                    concentration: 0.4, baseline: 0.4, release_rate: 0.0,
                    reuptake_rate: 0.6, decay_halflife_ms: 100.0,
                    receptors: HashMap::from([
                        ("alpha2".to_string(), ReceptorState {
                            sensitivity: 1.0, desensitization_rate: 0.012,
                            current_desensitization: 0.0, binding_affinity: 0.7,
                        }),
                    ]),
                }),
            ]),
            temporal_scale: TemporalScale::Fast,
        };

        let cortisol = NeuromodulatorSystem {
            regions: HashMap::from([
                ("systemic".to_string(), NeurotransmitterRegion {
                    concentration: 0.3, baseline: 0.3, release_rate: 0.0,
                    reuptake_rate: 0.05, decay_halflife_ms: 86_400_000.0,
                    receptors: HashMap::new(),
                }),
            ]),
            temporal_scale: TemporalScale::Slow,
        };

        let acetylcholine = NeuromodulatorSystem {
            regions: HashMap::from([
                ("hippocampus".to_string(), NeurotransmitterRegion {
                    concentration: 0.35, baseline: 0.35, release_rate: 0.0,
                    reuptake_rate: 0.5, decay_halflife_ms: 200.0,
                    receptors: HashMap::from([
                        ("M1".to_string(), ReceptorState {
                            sensitivity: 1.0, desensitization_rate: 0.008,
                            current_desensitization: 0.0, binding_affinity: 0.75,
                        }),
                    ]),
                }),
            ]),
            temporal_scale: TemporalScale::Medium,
        };

        let interaction_rules = vec![];

        NeurochemicalState {
            dopamine, serotonin, norepinephrine, cortisol, acetylcholine,
            interaction_rules, current_timestep_ms: 0,
        }
    }

    fn get_system_mut(&mut self, idx: usize) -> Option<&mut NeuromodulatorSystem> {
        match idx {
            0 => Some(&mut self.dopamine),
            1 => Some(&mut self.serotonin),
            2 => Some(&mut self.norepinephrine),
            3 => Some(&mut self.cortisol),
            4 => Some(&mut self.acetylcholine),
            _ => None,
        }
    }

    fn get_receptor_output(&self, system_idx: usize, region: &str, receptor: &str) -> f32 {
        let system = match system_idx {
            0 => &self.dopamine,
            1 => &self.serotonin,
            2 => &self.norepinephrine,
            3 => &self.cortisol,
            4 => &self.acetylcholine,
            _ => return 0.0,
        };
        
        system.regions.get(region)
            .and_then(|r| r.receptors.get(receptor))
            .map(|rec| {
                let conc = system.regions.get(region).unwrap().concentration;
                conc * rec.sensitivity * rec.binding_affinity
            })
            .unwrap_or(0.0)
    }

    fn phasic_release(&mut self, system_idx: usize, region: &str, amount: f32) {
        if let Some(sys) = self.get_system_mut(system_idx) {
            if let Some(r) = sys.regions.get_mut(region) {
                r.release_rate += amount;
            }
        }
    }

    fn step(&mut self, dt_ms: u64) {
        let dt = dt_ms as f32;
        
        for sys in [&mut self.dopamine, &mut self.serotonin, 
                    &mut self.norepinephrine, &mut self.cortisol, 
                    &mut self.acetylcholine].iter_mut() {
            for region in sys.regions.values_mut() {
                if region.release_rate != 0.0 {
                    region.concentration += region.release_rate * (dt / 1000.0);
                    region.release_rate *= 0.9f32.powf(dt / 100.0);
                }
                
                if region.decay_halflife_ms > 0.0 {
                    let k = std::f32::consts::LN_2 / region.decay_halflife_ms;
                    region.concentration *= (-k * dt).exp();
                }
                
                let reuptake = region.reuptake_rate * region.concentration * (dt / 1000.0);
                region.concentration = (region.concentration - reuptake).max(0.0);
                
                for rec in region.receptors.values_mut() {
                    let desens_inc = rec.desensitization_rate * region.concentration * (dt / 1000.0);
                    rec.current_desensitization = (rec.current_desensitization + desens_inc).min(1.0);
                    rec.current_desensitization = (rec.current_desensitization - 0.005 * dt / 1000.0).max(0.0);
                    rec.sensitivity = 1.0 - rec.current_desensitization * 0.8;
                }
                
                let drift = (region.baseline - region.concentration) * 0.001 * (dt / 1000.0);
                region.concentration += drift;
            }
        }
        
        self.current_timestep_ms += dt_ms;
    }

    fn compute_cognitive_modulation(&self) -> CognitiveModulation {
        let da_nacc = self.get_receptor_output(0, "nucleus_accumbens", "D1");
        let sero_pfc = self.get_receptor_output(1, "prefrontal_cortex", "5HT1A");
        let cortisol = self.cortisol.regions.get("systemic").map(|r| r.concentration).unwrap_or(0.3);
        let ne_lc = self.get_receptor_output(2, "locus_coeruleus", "alpha2");
        
        let learning_rate_multiplier = {
            let da_boost = da_nacc;
            let stress_penalty = (1.0 - cortisol.min(1.0) * 0.5).max(0.1);
            (da_boost * stress_penalty).clamp(0.1, 3.0)
        };
        
        let exploration_temperature = {
            let ne_drive = ne_lc * 2.0;
            let da_curiosity = da_nacc * 0.5;
            (ne_drive + da_curiosity).clamp(0.5, 3.0)
        };
        
        let risk_sensitivity = {
            let reward_seeking = da_nacc * 1.5;
            let harm_avoidance = (1.0 - sero_pfc) * 1.2;
            let stress = cortisol * 0.8;
            (reward_seeking - harm_avoidance - stress).clamp(-2.0, 2.0)
        };
        
        let emotional_valence = {
            let positive = da_nacc * 1.5;
            let negative = (1.0 - sero_pfc) * 1.0 + cortisol * 0.5;
            (positive - negative).clamp(-1.0, 1.0)
        };
        
        let temporal_discount = {
            let patience = (da_nacc * 0.4 + sero_pfc * 0.6).min(1.0);
            0.9 + patience * 0.09
        };
        
        CognitiveModulation {
            learning_rate_multiplier,
            exploration_temperature,
            risk_sensitivity,
            working_memory_capacity: 7.0,
            emotional_valence,
            temporal_discount,
            attention_focus: 1.0,
            impulse_control: 1.0,
        }
    }

    fn process_prediction_error(&mut self, td_error: f32) {
        if td_error > 0.1 {
            let burst_amount = (td_error * 2.0).min(1.0);
            self.phasic_release(0, "nucleus_accumbens", burst_amount);
        } else if td_error < -0.1 {
            let dip_amount = (td_error.abs() * 1.5).min(0.8);
            self.phasic_release(0, "nucleus_accumbens", -dip_amount);
        }
    }

    fn process_threat(&mut self, threat_level: f32) {
        if threat_level > 0.3 {
            self.phasic_release(3, "systemic", threat_level * 0.5);
            self.phasic_release(2, "locus_coeruleus", threat_level * 0.8);
        }
    }
}

impl GridWorld {
    fn new() -> Self {
        let mut threats = HashMap::new();
        threats.insert(State { x: 5, y: 5 }, 0.6);
        threats.insert(State { x: 3, y: 7 }, 0.4);
        
        let mut social_zones = HashMap::new();
        social_zones.insert(State { x: 1, y: 8 }, 0.4);
        social_zones.insert(State { x: 8, y: 1 }, 0.3);
        
        GridWorld {
            width: 10,
            height: 10,
            goal: State { x: 9, y: 9 },
            obstacles: vec![State { x: 5, y: 4 }, State { x: 5, y: 6 }],
            threats,
            social_zones,
        }
    }

    fn is_valid(&self, state: State) -> bool {
        state.x >= 0 && state.x < self.width &&
        state.y >= 0 && state.y < self.height &&
        !self.obstacles.contains(&state)
    }

    fn get_extrinsic_reward(&self, state: State) -> f32 {
        if state == self.goal { 10.0 } else { 0.0 }
    }

    fn get_threat(&self, state: State) -> f32 {
        *self.threats.get(&state).unwrap_or(&0.0)
    }

    fn get_social_value(&self, state: State) -> f32 {
        *self.social_zones.get(&state).unwrap_or(&0.0)
    }

    fn is_terminal(&self, state: State) -> bool {
        state == self.goal
    }
}

impl PersonalityAgent {
    fn new(start_state: State, world_width: i32, world_height: i32) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let state_dim = 2;
        let action_dim = 4;
        let base_learning_rate = 0.001;
        
        let actor_critic = ActorCritic::new(&vs.root(), state_dim, action_dim, base_learning_rate);
        
        PersonalityAgent {
            neurochemistry: NeurochemicalState::new(),
            homeostasis: HomeostaticState::new(),
            episodic_memory: EpisodicMemory::new(10000),
            theory_of_mind: TheoryOfMind::new(),
            actor_critic,
            state: start_state,
            base_learning_rate,
            base_exploration: 0.2,
            episode_count: 0,
            total_reward: 0.0,
            intrinsic_reward_weight: 0.3,
            social_status: 0.5,
            world_width,
            world_height,
        }
    }

    fn process_social_event(&mut self, quality: f32) {
        if quality == 0.0 { return; }
        self.homeostasis.social_interaction(quality);
        let status_change = (quality * 0.2).min(0.2);
        self.social_status = (self.social_status + status_change).clamp(0.0, 1.0);
        let status_benefit_signal = status_change * (1.0 - self.social_status);
        if status_benefit_signal > 0.01 {
            let burst_amount = (status_benefit_signal * 3.0).min(0.5);
            self.neurochemistry.phasic_release(1, "prefrontal_cortex", burst_amount);
        }
    }

    fn simulate_other_agent_step(&mut self, agent_id: usize, world: &GridWorld) -> (State, Action) {
        let model = self.theory_of_mind.other_agent_models.get(&agent_id);
        
        let current_state = if let Some(m) = model {
            m.current_state
        } else {
            State { x: 5, y: 5 }
        };

        let mut rng = rand::thread_rng();
        let action = if rng.gen::<f32>() < 0.7 {
            let actions = [Action::Up, Action::Down, Action::Left, Action::Right];
            actions[rng.gen_range(0..4)]
        } else {
            let goal = world.goal;
            let dx = goal.x - current_state.x;
            let dy = goal.y - current_state.y;
            
            if dx.abs() > dy.abs() {
                if dx > 0 { Action::Right } else { Action::Left }
            } else {
                if dy > 0 { Action::Down } else { Action::Up }
            }
        };

        let next_state = match action {
            Action::Up => State { x: current_state.x, y: current_state.y - 1 },
            Action::Down => State { x: current_state.x, y: current_state.y + 1 },
            Action::Left => State { x: current_state.x - 1, y: current_state.y },
            Action::Right => State { x: current_state.x + 1, y: current_state.y },
        };

        let valid_next_state = if world.is_valid(next_state) {
            next_state
        } else {
            current_state
        };

        self.theory_of_mind.observe_other_agent(agent_id, valid_next_state, action);

        (valid_next_state, action)
    }

    fn choose_action(&mut self, world: &GridWorld) -> Action {
        let cog_mod = self.neurochemistry.compute_cognitive_modulation();
        self.theory_of_mind.update_from_neurochemistry(&cog_mod);
        
        let temperature = cog_mod.exploration_temperature;
        
        let state_vec = self.state.to_vector(self.world_width, self.world_height);
        let (action_idx, _probs) = self.actor_critic.select_action(&state_vec, temperature);
        
        let mut selected_action = Action::from_index(action_idx);
        
        let other_agent_id = 1;
        let tom_capacity = self.theory_of_mind.mentalizing_capacity;
        
        if tom_capacity > 0.5 && self.social_status < 0.7 {
            if let Some(predicted_action) = self.theory_of_mind.predict_other_action(other_agent_id, self.state) {
                let should_cooperate = self.theory_of_mind.should_cooperate(other_agent_id);
                
                if should_cooperate {
                    let mut rng = rand::thread_rng();
                    let alignment_probability = tom_capacity * (1.0 - self.social_status) * 0.3;
                    
                    if rng.gen::<f32>() < alignment_probability {
                        selected_action = predicted_action;
                    }
                }
            }
        }
        
        selected_action
    }

    fn apply_action(&self, state: State, action: Action) -> State {
        match action {
            Action::Up => State { x: state.x, y: state.y - 1 },
            Action::Down => State { x: state.x, y: state.y + 1 },
            Action::Left => State { x: state.x - 1, y: state.y },
            Action::Right => State { x: state.x + 1, y: state.y },
        }
    }

    fn learn(&mut self, world: &GridWorld, action: Action, next_state: State, done: bool) {
        let extrinsic_reward = world.get_extrinsic_reward(next_state);
        
        let threat = world.get_threat(next_state);
        if threat > 0.0 {
            self.homeostasis.experience_threat(threat);
        }
        
        let social_quality = world.get_social_value(next_state);
        if social_quality > 0.0 {
            self.process_social_event(social_quality);
        }
        
        let total_reward = extrinsic_reward;
        
        let state_vec = self.state.to_vector(self.world_width, self.world_height);
        let next_state_vec = next_state.to_vector(self.world_width, self.world_height);
        
        let episode = Episode {
            state: state_vec,
            action: action.to_index(),
            reward: total_reward,
            next_state: next_state_vec,
            done,
            emotional_valence: 0.0,
            homeostatic_significance: self.homeostasis.get_homeostatic_error(),
            timestamp: self.neurochemistry.current_timestep_ms,
        };
        self.episodic_memory.store(episode);
        
        if self.episodic_memory.buffer.len() > 32 {
            let batch = self.episodic_memory.sample_batch(32);
            
            let states: Vec<Vec<f32>> = batch.iter().map(|e| e.state.clone()).collect();
            let actions: Vec<i64> = batch.iter().map(|e| e.action).collect();
            let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
            let next_states: Vec<Vec<f32>> = batch.iter().map(|e| e.next_state.clone()).collect();
            let dones: Vec<bool> = batch.iter().map(|e| e.done).collect();
            
            let cog_mod = self.neurochemistry.compute_cognitive_modulation();
            
            self.actor_critic.update(
                &states,
                &actions,
                &rewards,
                &next_states,
                &dones,
                cog_mod.temporal_discount,
                cog_mod.learning_rate_multiplier,
            );
        }
        
        self.neurochemistry.step(100);
        self.homeostasis.update(0.1);
        
        self.state = next_state;
        self.total_reward += total_reward;
    }

    fn run_episode(&mut self, world: &GridWorld, max_steps: usize) -> (usize, f32) {
        self.state = State { x: 0, y: 0 };
        let mut steps = 0;
        let episode_reward = self.total_reward;
        
        let other_agent_id = 1;
        
        self.theory_of_mind.other_agent_models.insert(other_agent_id, OtherAgentModel {
            current_state: State { x: 5, y: 5 },
            estimated_goals: vec![State { x: 5, y: 5 }],
            estimated_threat_sensitivity: 0.5,
            estimated_social_orientation: 0.8,
            confidence: 0.0,
            interaction_count: 0,
        });
        
        while steps < max_steps && !world.is_terminal(self.state) {
            let (other_agent_next_state, _other_agent_action) = 
                self.simulate_other_agent_step(other_agent_id, world);
            
            let action = self.choose_action(world);
            let next_state = self.apply_action(self.state, action);
            
            if !world.is_valid(next_state) {
                continue;
            }
            
            if next_state == other_agent_next_state {
                let interaction_reward = 0.3 * self.theory_of_mind.mentalizing_capacity;
                self.process_social_event(interaction_reward);
            }
            
            let done = world.is_terminal(next_state);
            self.learn(world, action, next_state, done);
            steps += 1;
        }
        
        self.episode_count += 1;
        (steps, self.total_reward - episode_reward)
    }
}

fn main() {
    let world = GridWorld::new();
    
    println!("=== Actor-Critic Personality Agent with Theory of Mind ===");
    let mut agent = PersonalityAgent::new(State { x: 0, y: 0 }, world.width, world.height);
    
    for episode in 0..500 {
        let (steps, reward) = agent.run_episode(&world, 100);
        
        if episode % 50 == 0 {
            let cog = agent.neurochemistry.compute_cognitive_modulation();
            println!("\nEpisode {}: steps={}, reward={:.2}", episode, steps, reward);
            println!("  Cognition: LR_mult={:.3}, Explore_temp={:.3}, Discount={:.3}",
                     cog.learning_rate_multiplier, 
                     cog.exploration_temperature,
                     cog.temporal_discount);
            println!("  Social: status={:.3}, ToM_capacity={:.3}",
                     agent.social_status,
                     agent.theory_of_mind.mentalizing_capacity);
            println!("  Homeostasis: safety={:.2}, social={:.2}, curiosity={:.2}",
                     agent.homeostasis.safety, 
                     agent.homeostasis.social_connection, 
                     agent.homeostasis.curiosity);
        }
    }
}
