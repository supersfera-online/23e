// ======================================================================
// ЭТАП 3: + GridWorld — масштабируемая среда
// Источник: main_with_tom.rs (Grid + threats + social zones)
// Сохранено: obstacles, threats, social_zones, goal
// Добавлено: параметрический размер (не хардкод 10x10)
// ======================================================================

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    pub fn to_index(self) -> i64 {
        match self {
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
        }
    }

    pub fn from_index(idx: i64) -> Self {
        match idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => Action::Up,
        }
    }

    pub const ACTION_DIM: i64 = 4;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn apply(self, action: Action) -> Self {
        match action {
            Action::Up => Position { x: self.x, y: self.y - 1 },
            Action::Down => Position { x: self.x, y: self.y + 1 },
            Action::Left => Position { x: self.x - 1, y: self.y },
            Action::Right => Position { x: self.x + 1, y: self.y },
        }
    }

    /// Нормализованный вектор состояния для RL
    pub fn to_state_vec(self, width: i32, height: i32) -> Vec<f32> {
        vec![
            self.x as f32 / width as f32,
            self.y as f32 / height as f32,
        ]
    }

    pub fn manhattan_distance(self, other: Position) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

#[derive(Clone, Debug)]
pub struct GridWorld {
    pub width: i32,
    pub height: i32,
    pub goal: Position,
    pub obstacles: Vec<Position>,
    pub threats: HashMap<Position, f32>,
    pub social_zones: HashMap<Position, f32>,
}

impl GridWorld {
    pub fn new(width: i32, height: i32) -> Self {
        let mut threats = HashMap::new();
        // Зоны угроз масштабируются с размером мира
        let mid_x = width / 2;
        let mid_y = height / 2;
        threats.insert(Position { x: mid_x, y: mid_y }, 0.6);
        threats.insert(Position { x: mid_x / 2, y: mid_y + mid_y / 2 }, 0.4);

        let mut social_zones = HashMap::new();
        social_zones.insert(Position { x: 1, y: height - 2 }, 0.4);
        social_zones.insert(Position { x: width - 2, y: 1 }, 0.3);

        GridWorld {
            width,
            height,
            goal: Position { x: width - 1, y: height - 1 },
            obstacles: vec![
                Position { x: mid_x, y: mid_y - 1 },
                Position { x: mid_x, y: mid_y + 1 },
            ],
            threats,
            social_zones,
        }
    }

    pub fn is_valid(&self, pos: Position) -> bool {
        pos.x >= 0
            && pos.x < self.width
            && pos.y >= 0
            && pos.y < self.height
            && !self.obstacles.contains(&pos)
    }

    pub fn get_extrinsic_reward(&self, pos: Position) -> f32 {
        if pos == self.goal { 10.0 } else { 0.0 }
    }

    pub fn get_threat(&self, pos: Position) -> f32 {
        self.threats.get(&pos).copied().unwrap_or(0.0)
    }

    pub fn get_social_value(&self, pos: Position) -> f32 {
        self.social_zones.get(&pos).copied().unwrap_or(0.0)
    }

    pub fn is_terminal(&self, pos: Position) -> bool {
        pos == self.goal
    }
}
