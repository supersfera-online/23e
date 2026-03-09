// ======================================================================
// ЭТАП 3: + Theory of Mind
// Источник: main_with_tom.rs (самый старый файл)
// Сохранено ВСЁ: предсказание действий, кооперация, confidence,
//   социальный статус, influence by neurochemistry
// Добавлено: интеграция с CognitiveModulation из neurochemistry.rs
// ======================================================================

use std::collections::HashMap;

use crate::neurochemistry::CognitiveModulation;
use crate::world::{Action, Position};

#[derive(Clone, Debug)]
pub struct OtherAgentModel {
    pub current_position: Position,
    pub estimated_goals: Vec<Position>,
    pub estimated_threat_sensitivity: f32,
    pub estimated_social_orientation: f32,
    pub confidence: f32,
    pub interaction_count: usize,
}

pub struct TheoryOfMind {
    pub other_agent_models: HashMap<usize, OtherAgentModel>,
    pub mentalizing_capacity: f32,
}

impl TheoryOfMind {
    pub fn new() -> Self {
        TheoryOfMind {
            other_agent_models: HashMap::new(),
            mentalizing_capacity: 1.0,
        }
    }

    /// Наблюдение за другим агентом: обновляет модель
    pub fn observe_other_agent(
        &mut self,
        agent_id: usize,
        observed_pos: Position,
        observed_action: Action,
    ) {
        let model = self
            .other_agent_models
            .entry(agent_id)
            .or_insert(OtherAgentModel {
                current_position: observed_pos,
                estimated_goals: Vec::new(),
                estimated_threat_sensitivity: 0.5,
                estimated_social_orientation: 0.5,
                confidence: 0.0,
                interaction_count: 0,
            });

        model.interaction_count += 1;
        model.confidence = (model.confidence + 0.1).min(1.0);
        model.current_position = observed_pos;

        let inferred_goal = observed_pos.apply(observed_action);

        if !model.estimated_goals.contains(&inferred_goal) {
            model.estimated_goals.push(inferred_goal);
        }

        if model.estimated_goals.len() > 5 {
            model.estimated_goals.remove(0);
        }
    }

    /// Предсказание действия другого агента на основе модели
    pub fn predict_other_action(
        &self,
        agent_id: usize,
    ) -> Option<Action> {
        let model = self.other_agent_models.get(&agent_id)?;

        if model.estimated_goals.is_empty() || model.confidence < 0.3 {
            return None;
        }

        let nearest_goal = model
            .estimated_goals
            .iter()
            .min_by_key(|goal| model.current_position.manhattan_distance(**goal))?;

        let dx = nearest_goal.x - model.current_position.x;
        let dy = nearest_goal.y - model.current_position.y;

        if dx.abs() > dy.abs() {
            if dx > 0 {
                Some(Action::Right)
            } else {
                Some(Action::Left)
            }
        } else if dy > 0 {
            Some(Action::Down)
        } else {
            Some(Action::Up)
        }
    }

    /// Решение о кооперации на основе модели другого агента
    pub fn should_cooperate(&self, agent_id: usize) -> bool {
        self.other_agent_models
            .get(&agent_id)
            .map_or(false, |model| {
                model.estimated_social_orientation > 0.6 && model.confidence > 0.4
            })
    }

    /// Обновление ToM на основе РЕАЛЬНОЙ когнитивной модуляции (из нейрохимии)
    pub fn update_from_neurochemistry(&mut self, cog_mod: &CognitiveModulation) {
        let serotonin_factor = (1.0 + cog_mod.emotional_valence).clamp(0.0, 2.0);
        let base_capacity = 0.5 + (serotonin_factor - 1.0) * 0.3;

        let working_memory_factor = (cog_mod.working_memory_capacity / 7.0).min(1.0);

        self.mentalizing_capacity = (base_capacity * working_memory_factor).clamp(0.1, 1.0);
    }
}

impl Default for TheoryOfMind {
    fn default() -> Self {
        Self::new()
    }
}
