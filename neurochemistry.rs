// ======================================================================
// ЭТАП 1: ОСНОВА — Богатая нейрохимическая модель
// Источник: main_with_tom.rs (самый старый файл)
// Сохранено ВСЁ: регионы мозга, рецепторы, десенсибилизация,
//   фармакодинамика, кросс-системные взаимодействия,
//   когнитивная модуляция
// ======================================================================

use std::collections::HashMap;

// --- Рецепторная модель ---

#[derive(Clone, Debug)]
pub struct ReceptorState {
    pub sensitivity: f32,
    pub desensitization_rate: f32,
    pub current_desensitization: f32,
    pub binding_affinity: f32,
}

// --- Нейротрансмиттерный регион ---

#[derive(Clone, Debug)]
pub struct NeurotransmitterRegion {
    pub concentration: f32,
    pub baseline: f32,
    pub release_rate: f32,
    pub reuptake_rate: f32,
    pub decay_halflife_ms: f32,
    pub receptors: HashMap<String, ReceptorState>,
}

// --- Временные масштабы нейромодуляции ---

#[derive(Clone, Debug)]
pub enum TemporalScale {
    Fast,   // дофамин, норэпинефрин — миллисекунды
    Medium, // серотонин, ацетилхолин — секунды
    Slow,   // кортизол — часы/дни
}

// --- Нейромодуляторная система (один нейротрансмиттер) ---

#[derive(Clone, Debug)]
pub struct NeuromodulatorSystem {
    pub regions: HashMap<String, NeurotransmitterRegion>,
    pub temporal_scale: TemporalScale,
}

// --- Кросс-системные взаимодействия ---

#[derive(Clone, Debug)]
pub enum ModulationType {
    Facilitation,
    Inhibition,
    SensitivityModulation,
    ReleaseModulation,
    ReceptorDensityChange,
}

#[derive(Clone, Debug)]
pub struct InteractionRule {
    pub source_system: usize,
    pub source_region: String,
    pub target_system: usize,
    pub target_region: String,
    pub modulation_type: ModulationType,
    pub strength: f32,
    pub threshold: f32,
    pub time_constant_ms: f32,
}

// --- Полное нейрохимическое состояние ---

pub struct NeurochemicalState {
    pub dopamine: NeuromodulatorSystem,
    pub serotonin: NeuromodulatorSystem,
    pub norepinephrine: NeuromodulatorSystem,
    pub cortisol: NeuromodulatorSystem,
    pub acetylcholine: NeuromodulatorSystem,
    pub interaction_rules: Vec<InteractionRule>,
    pub current_timestep_ms: u64,
}

// --- Когнитивная модуляция (выход нейрохимии → поведение) ---

#[derive(Clone, Debug)]
pub struct CognitiveModulation {
    pub learning_rate_multiplier: f32,
    pub exploration_temperature: f32,
    pub risk_sensitivity: f32,
    pub working_memory_capacity: f32,
    pub emotional_valence: f32,
    pub temporal_discount: f32,
    pub attention_focus: f32,
    pub impulse_control: f32,
}

impl NeurochemicalState {
    /// Создаёт полную нейрохимическую модель с 5 системами и регионами мозга
    pub fn new() -> Self {
        let dopamine = NeuromodulatorSystem {
            regions: HashMap::from([(
                "nucleus_accumbens".into(),
                NeurotransmitterRegion {
                    concentration: 0.4,
                    baseline: 0.4,
                    release_rate: 0.0,
                    reuptake_rate: 0.85,
                    decay_halflife_ms: 45.0,
                    receptors: HashMap::from([(
                        "D1".into(),
                        ReceptorState {
                            sensitivity: 1.1,
                            desensitization_rate: 0.018,
                            current_desensitization: 0.0,
                            binding_affinity: 0.85,
                        },
                    )]),
                },
            )]),
            temporal_scale: TemporalScale::Fast,
        };

        let serotonin = NeuromodulatorSystem {
            regions: HashMap::from([(
                "prefrontal_cortex".into(),
                NeurotransmitterRegion {
                    concentration: 0.6,
                    baseline: 0.6,
                    release_rate: 0.0,
                    reuptake_rate: 0.3,
                    decay_halflife_ms: 3_600_000.0,
                    receptors: HashMap::from([(
                        "5HT1A".into(),
                        ReceptorState {
                            sensitivity: 1.0,
                            desensitization_rate: 0.001,
                            current_desensitization: 0.0,
                            binding_affinity: 0.8,
                        },
                    )]),
                },
            )]),
            temporal_scale: TemporalScale::Medium,
        };

        let norepinephrine = NeuromodulatorSystem {
            regions: HashMap::from([(
                "locus_coeruleus".into(),
                NeurotransmitterRegion {
                    concentration: 0.4,
                    baseline: 0.4,
                    release_rate: 0.0,
                    reuptake_rate: 0.6,
                    decay_halflife_ms: 100.0,
                    receptors: HashMap::from([(
                        "alpha2".into(),
                        ReceptorState {
                            sensitivity: 1.0,
                            desensitization_rate: 0.012,
                            current_desensitization: 0.0,
                            binding_affinity: 0.7,
                        },
                    )]),
                },
            )]),
            temporal_scale: TemporalScale::Fast,
        };

        let cortisol = NeuromodulatorSystem {
            regions: HashMap::from([(
                "systemic".into(),
                NeurotransmitterRegion {
                    concentration: 0.3,
                    baseline: 0.3,
                    release_rate: 0.0,
                    reuptake_rate: 0.05,
                    decay_halflife_ms: 86_400_000.0,
                    receptors: HashMap::new(),
                },
            )]),
            temporal_scale: TemporalScale::Slow,
        };

        let acetylcholine = NeuromodulatorSystem {
            regions: HashMap::from([(
                "hippocampus".into(),
                NeurotransmitterRegion {
                    concentration: 0.35,
                    baseline: 0.35,
                    release_rate: 0.0,
                    reuptake_rate: 0.5,
                    decay_halflife_ms: 200.0,
                    receptors: HashMap::from([(
                        "M1".into(),
                        ReceptorState {
                            sensitivity: 1.0,
                            desensitization_rate: 0.008,
                            current_desensitization: 0.0,
                            binding_affinity: 0.75,
                        },
                    )]),
                },
            )]),
            temporal_scale: TemporalScale::Medium,
        };

        NeurochemicalState {
            dopamine,
            serotonin,
            norepinephrine,
            cortisol,
            acetylcholine,
            interaction_rules: vec![],
            current_timestep_ms: 0,
        }
    }

    /// Доступ к системе по индексу (0=DA, 1=5HT, 2=NE, 3=Cort, 4=ACh)
    pub fn get_system_mut(&mut self, idx: usize) -> Option<&mut NeuromodulatorSystem> {
        match idx {
            0 => Some(&mut self.dopamine),
            1 => Some(&mut self.serotonin),
            2 => Some(&mut self.norepinephrine),
            3 => Some(&mut self.cortisol),
            4 => Some(&mut self.acetylcholine),
            _ => None,
        }
    }

    pub fn get_system(&self, idx: usize) -> Option<&NeuromodulatorSystem> {
        match idx {
            0 => Some(&self.dopamine),
            1 => Some(&self.serotonin),
            2 => Some(&self.norepinephrine),
            3 => Some(&self.cortisol),
            4 => Some(&self.acetylcholine),
            _ => None,
        }
    }

    /// Рецепторный выход = концентрация × чувствительность × аффинность
    pub fn get_receptor_output(&self, system_idx: usize, region: &str, receptor: &str) -> f32 {
        let Some(system) = self.get_system(system_idx) else {
            return 0.0;
        };

        system
            .regions
            .get(region)
            .and_then(|r| {
                r.receptors.get(receptor).map(|rec| {
                    r.concentration * rec.sensitivity * rec.binding_affinity
                })
            })
            .unwrap_or(0.0)
    }

    /// Фазический выброс нейромедиатора в конкретном регионе
    pub fn phasic_release(&mut self, system_idx: usize, region: &str, amount: f32) {
        if let Some(sys) = self.get_system_mut(system_idx) {
            if let Some(r) = sys.regions.get_mut(region) {
                r.release_rate += amount;
            }
        }
    }

    /// Шаг временной динамики: распад, реаптейк, десенсибилизация, дрифт к baseline
    pub fn step(&mut self, dt_ms: u64) {
        let dt = dt_ms as f32;

        let systems: [&mut NeuromodulatorSystem; 5] = [
            &mut self.dopamine,
            &mut self.serotonin,
            &mut self.norepinephrine,
            &mut self.cortisol,
            &mut self.acetylcholine,
        ];

        for sys in systems {
            for region in sys.regions.values_mut() {
                // Фазический выброс
                if region.release_rate != 0.0 {
                    region.concentration += region.release_rate * (dt / 1000.0);
                    region.release_rate *= 0.9f32.powf(dt / 100.0);
                }

                // Экспоненциальный распад
                if region.decay_halflife_ms > 0.0 {
                    let k = std::f32::consts::LN_2 / region.decay_halflife_ms;
                    region.concentration *= (-k * dt).exp();
                }

                // Реаптейк
                let reuptake = region.reuptake_rate * region.concentration * (dt / 1000.0);
                region.concentration = (region.concentration - reuptake).max(0.0);

                // Десенсибилизация/ресенсибилизация рецепторов
                for rec in region.receptors.values_mut() {
                    let desens_inc =
                        rec.desensitization_rate * region.concentration * (dt / 1000.0);
                    rec.current_desensitization =
                        (rec.current_desensitization + desens_inc).min(1.0);
                    rec.current_desensitization =
                        (rec.current_desensitization - 0.005 * dt / 1000.0).max(0.0);
                    rec.sensitivity = 1.0 - rec.current_desensitization * 0.8;
                }

                // Дрифт к baseline
                let drift = (region.baseline - region.concentration) * 0.001 * (dt / 1000.0);
                region.concentration += drift;
            }
        }

        self.current_timestep_ms += dt_ms;
    }

    /// Конвертирует нейрохимическое состояние в когнитивные параметры
    pub fn compute_cognitive_modulation(&self) -> CognitiveModulation {
        let da_nacc = self.get_receptor_output(0, "nucleus_accumbens", "D1");
        let sero_pfc = self.get_receptor_output(1, "prefrontal_cortex", "5HT1A");
        let cortisol_level = self
            .cortisol
            .regions
            .get("systemic")
            .map_or(0.3, |r| r.concentration);
        let ne_lc = self.get_receptor_output(2, "locus_coeruleus", "alpha2");

        let learning_rate_multiplier = {
            let da_boost = da_nacc;
            let stress_penalty = (1.0 - cortisol_level.min(1.0) * 0.5).max(0.1);
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
            let stress = cortisol_level * 0.8;
            (reward_seeking - harm_avoidance - stress).clamp(-2.0, 2.0)
        };

        let emotional_valence = {
            let positive = da_nacc * 1.5;
            let negative = (1.0 - sero_pfc) * 1.0 + cortisol_level * 0.5;
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

    /// TD-ошибка → дофаминовый всплеск/провал (reward prediction error)
    pub fn process_prediction_error(&mut self, td_error: f32) {
        if td_error > 0.1 {
            let burst_amount = (td_error * 2.0).min(1.0);
            self.phasic_release(0, "nucleus_accumbens", burst_amount);
        } else if td_error < -0.1 {
            let dip_amount = (td_error.abs() * 1.5).min(0.8);
            self.phasic_release(0, "nucleus_accumbens", -dip_amount);
        }
    }

    /// Угроза → кортизол + норэпинефрин
    pub fn process_threat(&mut self, threat_level: f32) {
        if threat_level > 0.3 {
            self.phasic_release(3, "systemic", threat_level * 0.5);
            self.phasic_release(2, "locus_coeruleus", threat_level * 0.8);
        }
    }

    /// Удобные геттеры для упрощённого доступа к концентрациям
    pub fn dopamine_level(&self) -> f32 {
        self.dopamine
            .regions
            .get("nucleus_accumbens")
            .map_or(0.0, |r| r.concentration)
    }

    pub fn serotonin_level(&self) -> f32 {
        self.serotonin
            .regions
            .get("prefrontal_cortex")
            .map_or(0.0, |r| r.concentration)
    }

    pub fn cortisol_level(&self) -> f32 {
        self.cortisol
            .regions
            .get("systemic")
            .map_or(0.0, |r| r.concentration)
    }

    pub fn norepinephrine_level(&self) -> f32 {
        self.norepinephrine
            .regions
            .get("locus_coeruleus")
            .map_or(0.0, |r| r.concentration)
    }

    pub fn acetylcholine_level(&self) -> f32 {
        self.acetylcholine
            .regions
            .get("hippocampus")
            .map_or(0.0, |r| r.concentration)
    }
}
