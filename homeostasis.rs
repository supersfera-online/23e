// ======================================================================
// ЭТАП 1: ОСНОВА — Гомеостатический модуль
// Источник: main_with_tom.rs (самый старый файл)
// Сохранено ВСЁ: setpoints, homeostatic error, intrinsic motivation,
//   threat/social/explore events
// ======================================================================

#[derive(Clone, Debug)]
pub struct HomeostaticState {
    pub safety: f32,
    pub safety_setpoint: f32,
    pub social_connection: f32,
    pub social_setpoint: f32,
    pub curiosity: f32,
    pub curiosity_decay: f32,
}

impl HomeostaticState {
    pub fn new() -> Self {
        HomeostaticState {
            safety: 1.0,
            safety_setpoint: 0.8,
            social_connection: 0.5,
            social_setpoint: 0.6,
            curiosity: 1.0,
            curiosity_decay: 0.998,
        }
    }

    /// Пассивная динамика: безопасность восстанавливается, связь убывает, любопытство затухает
    pub fn update(&mut self, dt: f32) {
        self.safety = (self.safety + 0.01 * dt).min(1.0);
        self.social_connection -= 0.005 * dt;
        self.social_connection = self.social_connection.clamp(0.0, 1.0);
        self.curiosity *= self.curiosity_decay.powf(dt);
    }

    /// Ошибка гомеостаза: отклонение от setpoints (чем больше — тем сильнее мотивация)
    pub fn get_homeostatic_error(&self) -> f32 {
        let safety_error = (self.safety_setpoint - self.safety).max(0.0);
        let social_error = (self.social_setpoint - self.social_connection).max(0.0);
        safety_error * 1.5 + social_error * 0.8
    }

    /// Внутренняя мотивация = гомеостатический драйв + любопытство × новизна
    pub fn get_intrinsic_motivation(&self, state_novelty: f32) -> f32 {
        let homeostatic_drive = self.get_homeostatic_error();
        let curiosity_drive = self.curiosity * state_novelty;
        homeostatic_drive + curiosity_drive * 0.5
    }

    /// Переживание угрозы: снижает безопасность
    pub fn experience_threat(&mut self, intensity: f32) {
        self.safety -= intensity;
        self.safety = self.safety.max(0.0);
    }

    /// Социальное взаимодействие: повышает связь и обновляет любопытство
    pub fn social_interaction(&mut self, quality: f32) {
        self.social_connection = (self.social_connection + quality).min(1.0);
        self.curiosity = (self.curiosity + 0.05).min(1.0);
    }

    /// Успешное исследование: всплеск любопытства
    pub fn explore_success(&mut self) {
        self.curiosity = (self.curiosity + 0.1).min(1.0);
    }
}

impl Default for HomeostaticState {
    fn default() -> Self {
        Self::new()
    }
}
