use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use crate::{NeurochemicalState, HomeostaticState};

#[derive(Debug, Clone)]
pub struct ValueRule {
    pub name: &'static str,
    pub preconditions: &'static [&'static str],
    pub effects: &'static [&'static str],
    pub mass: f32,
}

impl ValueRule {
    pub const fn new(
        name: &'static str,
        preconditions: &'static [&'static str],
        effects: &'static [&'static str],
        mass: f32,
    ) -> Self {
        ValueRule { name, preconditions, effects, mass }
    }
}

#[derive(Clone)]
struct SearchNode {
    priority: f32,
    time_elapsed: f32,
    plan: Vec<&'static str>,
    goals: HashSet<String>,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.priority.to_bits() == other.priority.to_bits()
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.partial_cmp(&self.priority).unwrap_or(Ordering::Equal)
    }
}

pub struct PfcPlanner {
    rules: Vec<ValueRule>,
}

impl PfcPlanner {
    pub fn new(rules: Vec<ValueRule>) -> Self {
        PfcPlanner { rules }
    }

    pub fn default_personality() -> Self {
        PfcPlanner::new(vec![
            ValueRule::new("meet_safety_need",    &["unsafe"],          &["safety_met"],       2.0),
            ValueRule::new("seek_social_bond",    &["isolated"],        &["social_met"],       1.5),
            ValueRule::new("reduce_stress",       &["under_threat"],    &["stress_reduced"],   2.5),
            ValueRule::new("explore_novelty",     &["curious", "safe"], &["knowledge_gained"], 1.0),
            ValueRule::new("exploit_reward",      &["reward_signal"],   &["reward_obtained"],  1.0),
            ValueRule::new("cooperate",           &["social_met"],      &["cooperation_done"], 1.2),
            ValueRule::new("consolidate_memory",  &["knowledge_gained"]                      , &["experience_stored"], 0.8),
            ValueRule::new("restore_homeostasis", &["stress_reduced", "safety_met"], &["homeostasis_stable"], 3.0),
        ])
    }

    pub fn derive_world_state(
        neurochem: &NeurochemicalState,
        homeostasis: &HomeostaticState,
    ) -> HashSet<String> {
        let mut state = HashSet::new();

        if homeostasis.safety < 0.3 {
            state.insert("unsafe".to_string());
        } else {
            state.insert("safe".to_string());
        }

        if homeostasis.social_connection < 0.3 {
            state.insert("isolated".to_string());
        } else {
            state.insert("social_met".to_string());
        }

        if neurochem.cortisol > 0.6 {
            state.insert("under_threat".to_string());
        } else {
            state.insert("stress_reduced".to_string());
        }

        if homeostasis.curiosity > 0.5 {
            state.insert("curious".to_string());
        }

        if neurochem.dopamine > 0.6 {
            state.insert("reward_signal".to_string());
        }

        if homeostasis.safety >= 0.3 && neurochem.cortisol <= 0.6 {
            state.insert("safety_met".to_string());
        }

        state
    }

    pub fn plan(
        &self,
        target: &[&str],
        world_state: &HashSet<String>,
    ) -> Option<Vec<&'static str>> {
        let target_goals: HashSet<String> = target.iter().map(|s| s.to_string()).collect();

        let initial_pressure = (target_goals.difference(world_state).count()) as f32;

        let start = SearchNode {
            priority: initial_pressure,
            time_elapsed: 0.0,
            plan: vec![],
            goals: target_goals,
        };

        let mut frontier = BinaryHeap::new();
        frontier.push(start);

        let mut seen: HashSet<Vec<String>> = HashSet::new();

        while let Some(node) = frontier.pop() {
            let remaining: HashSet<String> = node.goals.difference(world_state).cloned().collect();

            if remaining.is_empty() {
                let mut plan = node.plan.clone();
                plan.reverse();
                return Some(plan);
            }

            let mut sig: Vec<String> = remaining.iter().cloned().collect();
            sig.sort();
            if seen.contains(&sig) {
                continue;
            }
            seen.insert(sig);

            for rule in &self.rules {
                let rule_effects: HashSet<String> = rule.effects.iter().map(|s| s.to_string()).collect();
                if rule_effects.is_disjoint(&remaining) {
                    continue;
                }

                let next_goals: HashSet<String> = remaining
                    .difference(&rule_effects)
                    .cloned()
                    .chain(rule.preconditions.iter().map(|s| s.to_string()))
                    .filter(|g| !world_state.contains(g))
                    .collect();

                let new_time = node.time_elapsed + 1.0;
                let entropic_pressure = next_goals.len() as f32 + new_time / rule.mass.max(0.01);

                let mut new_plan = node.plan.clone();
                new_plan.push(rule.name);

                frontier.push(SearchNode {
                    priority: entropic_pressure,
                    time_elapsed: new_time,
                    plan: new_plan,
                    goals: next_goals,
                });
            }
        }

        None
    }

    pub fn current_top_goal(
        &self,
        neurochem: &NeurochemicalState,
        homeostasis: &HomeostaticState,
    ) -> &'static str {
        if homeostasis.safety < 0.3 || neurochem.cortisol > 0.7 {
            return "homeostasis_stable";
        }
        if homeostasis.social_connection < 0.3 {
            return "cooperation_done";
        }
        if homeostasis.curiosity > 0.6 && neurochem.dopamine > 0.5 {
            return "experience_stored";
        }
        "homeostasis_stable"
    }

    pub fn evaluate_action(
        &self,
        action: i64,
        neurochem: &NeurochemicalState,
        homeostasis: &HomeostaticState,
        action_social_cost: f32,
    ) -> bool {
        if neurochem.cortisol > 0.8 && action == 3 {
            return false;
        }
        if homeostasis.safety < 0.2 && action_social_cost > 0.5 {
            return false;
        }
        if neurochem.serotonin < 0.2 && action_social_cost < -0.3 {
            return false;
        }
        true
    }

    pub fn gate_action(
        &self,
        proposed: i64,
        action_dim: i64,
        neurochem: &NeurochemicalState,
        homeostasis: &HomeostaticState,
        action_social_costs: &HashMap<i64, f32>,
    ) -> i64 {
        let cost = action_social_costs.get(&proposed).copied().unwrap_or(0.0);
        if self.evaluate_action(proposed, neurochem, homeostasis, cost) {
            return proposed;
        }

        for alt in 0..action_dim {
            if alt == proposed {
                continue;
            }
            let alt_cost = action_social_costs.get(&alt).copied().unwrap_or(0.0);
            if self.evaluate_action(alt, neurochem, homeostasis, alt_cost) {
                return alt;
            }
        }

        proposed
    }
}
