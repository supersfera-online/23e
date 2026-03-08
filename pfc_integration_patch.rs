// ---- 1. Add to PersonalityAgent struct ----

pub struct PersonalityAgent {
    // ... existing fields ...
    pub pfc_planner: PfcPlanner,
    pub action_social_costs: HashMap<i64, f32>,
}

// ---- 2. In PersonalityAgent::new() ----

PersonalityAgent {
    // ... existing fields ...
    pfc_planner: PfcPlanner::default_personality(),
    action_social_costs: {
        let mut m = HashMap::new();
        // action 0: move/explore -> neutral
        m.insert(0, 0.0_f32);
        // action 1: approach social -> positive
        m.insert(1, 0.3_f32);
        // action 2: withdraw -> mildly negative social
        m.insert(2, -0.1_f32);
        // action 3: aggressive/risky -> high social cost
        m.insert(3, -0.6_f32);
        m
    },
}

// ---- 3. In choose_action(), after action is selected, before return ----

if self.use_hierarchical || self.use_meta_learning {
    action = self.pfc_planner.gate_action(
        action,
        ACTION_DIM,
        &self.neurochemistry,
        &self.homeostasis,
        &self.action_social_costs,
    );
}

// ---- 4. Optional: derive world state and run planner to get active goal ----

let world_state = PfcPlanner::derive_world_state(&self.neurochemistry, &self.homeostasis);
let top_goal = self.pfc_planner.current_top_goal(&self.neurochemistry, &self.homeostasis);
if let Some(plan) = self.pfc_planner.plan(&[top_goal], &world_state) {
    // plan[0] is the next qualitative step — can be logged or used to bias skill selection
    let _ = plan;
}
