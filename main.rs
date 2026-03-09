// ======================================================================
// ЭТАПЫ 1-7: Полная интеграция
// Все модули работают ВМЕСТЕ, ничего не потеряно
//
//   neurochemistry.rs  — богатая нейрохимия (регионы, рецепторы)
//   homeostasis.rs     — гомеостаз (setpoints, drives)
//   memory.rs          — эпизодическая память
//   pfc_planner.rs     — GOAP на реальной нейрохимии
//   world.rs           — GridWorld
//   theory_of_mind.rs  — ToM из CognitiveModulation
//   actor_critic.rs    — нейросетевой RL (tch)
//   hrl.rs             — MetaController + Skills
//   meta_learner.rs    — GRU RL²
//   world_model.rs     — VAE world model
//   agent.rs           — PersonalityAgent (enum dispatch)
// ======================================================================

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod actor_critic;    // Этап 4
mod agent;           // Этап 7
mod homeostasis;     // Этап 1
mod hrl;             // Этап 5
mod memory;          // Этап 1
mod meta_learner;    // Этап 5
mod neurochemistry;  // Этап 1
mod pfc_planner;     // Этап 2
mod theory_of_mind;  // Этап 3
mod world;           // Этап 3
mod world_model;     // Этап 6

use std::collections::HashMap;
use std::time::Instant;

use eyre::Result;
use tch::Device;

use agent::{AgentConfig, PersonalityAgent};
use world::{Action, GridWorld, Position};

// --- Конфигурация тренировки ---

const NUM_AGENTS: usize = 5;
const NUM_STEPS: usize = 50_000;
const LOG_INTERVAL: usize = 1_000;
const SAVE_INTERVAL: usize = 10_000;
const WORLD_W: i32 = 20;
const WORLD_H: i32 = 20;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // GPU auto-detect (не хардкод R9700 — железо поменялось)
    let device = Device::cuda_if_available();
    println!("=== Neurochemical Personality Simulation ===");
    println!("Device: {:?}", device);
    println!(
        "Config: {} agents, {} steps, {}x{} world",
        NUM_AGENTS, NUM_STEPS, WORLD_W, WORLD_H
    );

    std::fs::create_dir_all("checkpoints")?;

    let config = AgentConfig::default();
    let grid = GridWorld::new(WORLD_W, WORLD_H);

    // Создаём агентов с разными стратегиями (enum dispatch)
    let mut agents: Vec<PersonalityAgent> = (0..NUM_AGENTS)
        .map(|i| match i % 3 {
            0 => PersonalityAgent::new_hierarchical(i, device, &config),
            1 => PersonalityAgent::new_meta_learning(i, device, &config),
            _ => PersonalityAgent::new_actor_critic(i, device, &config),
        })
        .collect();

    // Загрузка чекпоинтов
    if std::env::var("LOAD_CHECKPOINT").is_ok() {
        println!("Loading checkpoints...");
        for agent in &mut agents {
            let path = format!("checkpoints/agent_{}_final.pt", agent.id);
            if std::path::Path::new(&path).exists() {
                agent.vs.load(&path)?;
                println!("  Loaded: agent_{}", agent.id);
            }
        }
    }

    println!("\nAgents:");
    for agent in &agents {
        let tom = if agent.use_tom { "+ToM" } else { "" };
        println!("  Agent {}: {}{}", agent.id, agent.decision_type_name(), tom);
    }

    let start_time = Instant::now();
    let mut total_social_events: usize = 0;

    // --- Тренировочный цикл ---
    for step in 0..NUM_STEPS {
        // 1. Все агенты выбирают действия
        let mut actions: Vec<Action> = Vec::with_capacity(NUM_AGENTS);
        let mut positions: Vec<Position> = agents.iter().map(|a| a.position).collect();

        for agent in &mut agents {
            actions.push(agent.choose_action(&grid));
        }

        // 2. Применяем действия, обрабатываем шаги
        for i in 0..NUM_AGENTS {
            let next_pos = positions[i].apply(actions[i]);
            let valid_pos = if grid.is_valid(next_pos) {
                next_pos
            } else {
                positions[i]
            };

            agents[i].process_step(&grid, actions[i], valid_pos);
            positions[i] = valid_pos;
        }

        // 3. ToM: агенты наблюдают друг за другом
        for i in 0..NUM_AGENTS {
            for j in 0..NUM_AGENTS {
                if i != j {
                    agents[i].theory_of_mind.observe_other_agent(
                        j,
                        positions[j],
                        actions[j],
                    );
                }
            }
        }

        // 4. Социальные взаимодействия (встречи)
        for i in 0..NUM_AGENTS {
            for j in (i + 1)..NUM_AGENTS {
                if positions[i] == positions[j]
                    || positions[i].manhattan_distance(positions[j]) <= 1
                {
                    let quality = 0.2 + agents[i].social_status * 0.1;
                    agents[i].process_social_event(quality);
                    agents[j].process_social_event(quality);
                    total_social_events += 1;
                }
            }
        }

        // --- Логирование ---
        if step % LOG_INTERVAL == 0 || step == NUM_STEPS - 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let speed = (step + 1) as f32 / elapsed;
            let eta = (NUM_STEPS - step) as f32 / speed / 3600.0;

            println!(
                "\n=== STEP {}/{} ({:.1}%) | {:.0} steps/s | ETA: {:.1}h ===",
                step,
                NUM_STEPS,
                step as f32 / NUM_STEPS as f32 * 100.0,
                speed,
                eta
            );

            // Средние по популяции
            let n = agents.len() as f32;
            let avg_da: f32 = agents.iter().map(|a| a.neurochemistry.dopamine_level()).sum::<f32>() / n;
            let avg_5ht: f32 = agents.iter().map(|a| a.neurochemistry.serotonin_level()).sum::<f32>() / n;
            let avg_cort: f32 = agents.iter().map(|a| a.neurochemistry.cortisol_level()).sum::<f32>() / n;
            let avg_safety: f32 = agents.iter().map(|a| a.homeostasis.safety).sum::<f32>() / n;
            let avg_social: f32 = agents.iter().map(|a| a.homeostasis.social_connection).sum::<f32>() / n;

            println!(
                "Pop avg: DA={:.3} 5HT={:.3} Cort={:.3} | Safety={:.3} Social={:.3} | Events: {}",
                avg_da, avg_5ht, avg_cort, avg_safety, avg_social, total_social_events
            );

            for agent in &agents {
                let skill = agent
                    .current_skill_id()
                    .map_or("N".into(), |s| s.to_string());
                println!(
                    "  A{}: {} pos=({},{}) DA={:.2} 5HT={:.2} Cort={:.2} S={}",
                    agent.id,
                    agent.decision_type_name(),
                    agent.position.x,
                    agent.position.y,
                    agent.neurochemistry.dopamine_level(),
                    agent.neurochemistry.serotonin_level(),
                    agent.neurochemistry.cortisol_level(),
                    skill,
                );
            }
        }

        // --- Чекпоинтинг ---
        if step % SAVE_INTERVAL == 0 && step > 0 {
            println!("\nSaving checkpoints at step {}...", step);
            for agent in &agents {
                let path = format!("checkpoints/agent_{}_step_{}.pt", agent.id, step);
                agent.vs.save(&path).unwrap_or_else(|e| {
                    eprintln!("  Failed agent {}: {}", agent.id, e);
                });
            }
        }
    }

    // --- Финал ---
    let total_time = start_time.elapsed().as_secs_f32();
    println!("\n=== TRAINING COMPLETE: {} steps in {:.1}s ===", NUM_STEPS, total_time);

    for agent in &agents {
        let path = format!("checkpoints/agent_{}_final.pt", agent.id);
        agent.vs.save(&path)?;
        println!("  Saved: {}", path);
    }

    Ok(())
}
