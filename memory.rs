// ======================================================================
// ЭТАП 1: ОСНОВА — Эпизодическая память
// Источник: main_with_tom.rs (самый старый файл)
// Сохранено: кольцевой буфер, emotional valence, homeostatic significance
// ======================================================================

use std::collections::VecDeque;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Episode {
    pub state: Vec<f32>,
    pub action: i64,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub emotional_valence: f32,
    pub homeostatic_significance: f32,
    pub timestamp: u64,
}

pub struct EpisodicMemory {
    buffer: VecDeque<Episode>,
    max_size: usize,
}

impl EpisodicMemory {
    pub fn new(max_size: usize) -> Self {
        EpisodicMemory {
            buffer: VecDeque::with_capacity(max_size.min(65536)),
            max_size,
        }
    }

    pub fn store(&mut self, episode: Episode) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(episode);
    }

    pub fn sample_batch(&self, batch_size: usize) -> Vec<Episode> {
        let mut rng = rand::thread_rng();
        let actual_size = batch_size.min(self.buffer.len());
        let mut samples = Vec::with_capacity(actual_size);

        for _ in 0..actual_size {
            let idx = rng.gen_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }

        samples
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}
