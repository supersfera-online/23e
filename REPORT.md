# Neurochemical Personality Simulation — Полный отчёт

Дата: 2026-03-09

---

## 1. Анализ исходных файлов

### Хронология (от старого к новому)

| # | Файл | Суть | Проблема |
|---|------|------|----------|
| 1 | `main_with_tom.rs` | **ОСНОВА**. Богатая нейрохимия: 5 систем × регионы мозга × рецепторы с десенсибилизацией. ActorCritic, ToM, GridWorld 10×10, 1 агент, 500 эпизодов | — |
| 2 | `pfc_planner.rs` | GOAP планировщик (A* + entropic pressure), 8 value rules, action gating | — |
| 3 | `personality_rl_planner.py` | SupersphereEngine — оптимизатор порядка установки (A* планировщик для build pipeline) | Python — нарушает правило "No Python in ML pipeline" |
| 4 | `PFC_INTEGRATION_INSTRUCTIONS.txt` | Инструкции как подключить pfc_planner | — |
| 5 | `FINAL_INTEGRATION_with_pfc.rs` | Интеграция всех компонентов. **ПРОБЛЕМА: ПЕРЕПИСАЛА нейрохимию** — 5 систем с регионами/рецепторами заменены на 7 голых f32 полей. Добавлены HRL, MetaLearner, WorldModel, но фундамент потерян | **Потеря основы** |
| 6 | `FINAL_INTEGRATION_extended.rs` | Масштабирование переписанного: 5 агентов, 50k шагов, 20×20, чекпоинтинг | Наследует проблему |
| 7 | `FINAL_INTEGRATION_50M.rs` | Extreme scale переписанного: 100 агентов, 50M шагов, 200×200, GPU | Наследует проблему |
| 8 | Текстовые конфиги | `HIGH_PERFORMANCE_CONFIG.txt`, `EXTREME_SCALE_CONFIG.txt`, `EXTENDED_TRAINING_GUIDE.txt`, `RUN_50M_GUIDE.txt`, `QUICK_START.txt`, `INSTALLATION_OPTIMIZATION_ANALYSIS.txt`, `README.md` | R9700/gfx1201 — подтверждено rocminfo, конфиги актуальны |

### Что было потеряно при переписывании

```
main_with_tom.rs (ОСНОВА)          FINAL_INTEGRATION_with_pfc.rs
─────────────────────────           ─────────────────────────────
NeurochemicalState {                NeurochemicalState {
  dopamine: NeuromodulatorSystem {    dopamine: f32,        ← ЗАГЛУШКА
    regions: {                        serotonin: f32,       ← ЗАГЛУШКА
      "nucleus_accumbens": {          cortisol: f32,        ← ЗАГЛУШКА
        concentration: 0.4,           norepinephrine: f32,  ← ЗАГЛУШКА
        baseline: 0.4,                acetylcholine: f32,   ← ЗАГЛУШКА
        reuptake_rate: 0.85,          exploration_temp: f32, ← ЗАГЛУШКА
        decay_halflife_ms: 45.0,      impulse_control: f32, ← ЗАГЛУШКА
        receptors: {                }
          "D1": ReceptorState {
            sensitivity: 1.1,
            desensitization_rate: 0.018,
            binding_affinity: 0.85,
          }
        }
      }
    }
  },
  serotonin: NeuromodulatorSystem { ... },
  norepinephrine: ...,
  cortisol: ...,
  acetylcholine: ...,
  interaction_rules: Vec<InteractionRule>,
}

CognitiveModulation {               ← ПОЛНОСТЬЮ ПОТЕРЯНО
  learning_rate_multiplier,
  exploration_temperature,
  risk_sensitivity,
  working_memory_capacity,
  emotional_valence,
  temporal_discount,
  attention_focus,
  impulse_control,
}
```

**Потеряно:**
- Регионы мозга (nucleus_accumbens, prefrontal_cortex, locus_coeruleus, hippocampus)
- Рецепторы с десенсибилизацией (D1, 5HT1A, alpha2, M1)
- Фармакодинамика (reuptake, exponential decay, baseline drift, phasic release)
- CognitiveModulation (8 параметров, вычисляемых из рецепторных выходов)
- Связь нейрохимия → поведение через рецепторный выход

---

## 2. Реальное железо

### Проверено через `rocminfo` (2026-03-09)

```
CPU: AMD Ryzen 9 9950X (16 cores / 32 threads)
RAM: 172 GB
GPU 1 (дискретная): AMD Radeon AI PRO R9700 — Chip ID 0x7551, gfx1201, 64 CU, ~30 GB VRAM, 2350 MHz
GPU 2 (встроенная): AMD Radeon Graphics (iGPU 9950X) — Chip ID 0x13c0, gfx1201, 2 CU, shared RAM, 2200 MHz
ROCm runtime: 1.18
```

**Итог:**
- Обе GPU определяются как **gfx1201** — `HSA_OVERRIDE_GFX_VERSION=12.0.1` корректен
- GPU 1 (0x7551) — это **R9700**, старые конфиги были правильными
- GPU 2 (0x13c0) — встроенная графика процессора, 2 CU — для вычислений не используется

---

## 3. Новая архитектура (собрано правильно)

### Принцип: ВМЕСТЕ, а не ВМЕСТО

Каждый этап **добавляет** новый файл-модуль. Предыдущие модули не трогаются.

### Структура проекта

```
/media/slava/B/AIAI/
├── Cargo.toml
├── src/
│   ├── neurochemistry.rs   ← Этап 1: ОСНОВА (регионы, рецепторы, фармакодинамика)
│   ├── homeostasis.rs      ← Этап 1: гомеостаз (setpoints, drives, intrinsic motivation)
│   ├── memory.rs           ← Этап 1: эпизодическая память (кольцевой буфер)
│   ├── pfc_planner.rs      ← Этап 2: GOAP планировщик на РЕАЛЬНОЙ нейрохимии
│   ├── world.rs            ← Этап 3: GridWorld (масштабируемый, threats, social zones)
│   ├── theory_of_mind.rs   ← Этап 3: ToM (управляется CognitiveModulation)
│   ├── gpu.rs              ← Этап 4: XLA/PJRT обёртка — Rust FFI к XLA C API, R9700 через ROCm
│   ├── actor_critic.rs     ← Этап 4: 3-слойный Actor + Critic (Rust + XLA на GPU)
│   ├── hrl.rs              ← Этап 5: MetaController + Skills (HRL, XLA на GPU)
│   ├── meta_learner.rs     ← Этап 5: GRU-based RL² (Rust + XLA на GPU)
│   ├── world_model.rs      ← Этап 6: VAE encoder/decoder + GRU динамика (Rust + XLA на GPU)
│   ├── agent.rs            ← Этап 7: PersonalityAgent (enum dispatch, всё вместе)
│   └── main.rs             ← Этап 7: мульти-агент, GPU init, чекпоинтинг
├── (старые файлы — не тронуты)
│   ├── main_with_tom.rs
│   ├── pfc_planner.rs (корневой)
│   ├── FINAL_INTEGRATION_*.rs
│   └── *.txt
```

### Граф зависимостей модулей

```
neurochemistry.rs ──┐
                    ├──► pfc_planner.rs ──┐
homeostasis.rs ─────┤                    │
                    ├──► theory_of_mind.rs┤
memory.rs ──────────┤                    │
                    │                    ├──► agent.rs ──► main.rs
world.rs ───────────┤                    │
                    │       gpu.rs ──────┤  (XLA/PJRT → R9700)
                    │         │          │
actor_critic.rs ────┼─────────┤──────────┤
                    │         │          │
hrl.rs ─────────────┼─────────┤──────────┤
                    │         │          │
meta_learner.rs ────┼─────────┤──────────┤
                    │         │          │
world_model.rs ─────┴─────────┘──────────┘
```

---

## 4. Детали каждого этапа

### Этап 1: Фундамент (из main_with_tom.rs)

**neurochemistry.rs** — 270 строк
- `ReceptorState` — sensitivity, desensitization_rate, binding_affinity
- `NeurotransmitterRegion` — concentration, baseline, reuptake_rate, decay_halflife_ms, receptors
- `NeuromodulatorSystem` — regions (HashMap), temporal_scale
- `NeurochemicalState` — 5 систем: dopamine (nucleus_accumbens/D1), serotonin (prefrontal_cortex/5HT1A), norepinephrine (locus_coeruleus/alpha2), cortisol (systemic), acetylcholine (hippocampus/M1)
- `CognitiveModulation` — 8 параметров, вычисляемых из рецепторных выходов
- Методы: `step()` (динамика), `phasic_release()`, `process_prediction_error()`, `process_threat()`, `compute_cognitive_modulation()`
- Удобные геттеры: `dopamine_level()`, `serotonin_level()`, etc.

**homeostasis.rs** — 75 строк
- `HomeostaticState` — safety, social_connection, curiosity с setpoints
- Методы: `update()` (пассивная динамика), `get_homeostatic_error()`, `get_intrinsic_motivation()`, `experience_threat()`, `social_interaction()`, `explore_success()`

**memory.rs** — 65 строк
- `Episode` — state, action, reward, next_state, done, emotional_valence, homeostatic_significance, timestamp
- `EpisodicMemory` — VecDeque кольцевой буфер, `store()`, `sample_batch()`

**Проверка**: этапы 1-3 скомпилированы и запущены. Нейрохимическая динамика работает: кортизол растёт от угроз, дофамин истощается, homeostatic error нарастает.

### Этап 2: PFC Planner (из pfc_planner.rs)

**pfc_planner.rs** — 215 строк
- `ValueRule` — GOAP правила (preconditions, effects, mass)
- `PfcPlanner` — A* поиск с entropic pressure heuristic
- `derive_world_state()` — **принимает `&NeurochemicalState` и `&HomeostaticState`** (богатую нейрохимию, а не заглушки)
- `current_top_goal()` — приоритетная цель из реальной нейрохимии
- `gate_action()` — вето/замена действий на основе cortisol/serotonin levels
- 8 value rules: meet_safety_need, seek_social_bond, reduce_stress, explore_novelty, exploit_reward, cooperate, consolidate_memory, restore_homeostasis

**Проверка**: PFC планирует адаптивно. При cortisol > 0.7 и safety < 0.3 план усложняется: `["meet_safety_need", "reduce_stress", "restore_homeostasis"]`.

### Этап 3: World + Theory of Mind (из main_with_tom.rs)

**world.rs** — 115 строк
- `Action` enum (Up/Down/Left/Right) с `to_index()`/`from_index()`
- `Position` — координаты, `apply()`, `to_state_vec()`, `manhattan_distance()`
- `GridWorld` — параметрический размер, obstacles, threats (HashMap), social_zones

**theory_of_mind.rs** — 115 строк
- `OtherAgentModel` — position, estimated_goals, threat_sensitivity, social_orientation, confidence
- `TheoryOfMind` — HashMap моделей других агентов, mentalizing_capacity
- `observe_other_agent()` — обновление модели по наблюдениям
- `predict_other_action()` — предсказание действий по estimated_goals
- `should_cooperate()` — решение о кооперации
- `update_from_neurochemistry()` — **mentalizing_capacity вычисляется из CognitiveModulation** (emotional_valence, working_memory_capacity)

**Проверка**: ToM предсказывает действия (Some(Up)), mentalizing_capacity падает с 0.33→0.20 при нарастающем кортизоле.

### Этап 4: GPU-слой + Actor-Critic (Rust + XLA)

**gpu.rs** — Rust FFI к XLA C API
- `XlaDevice` — инициализация PJRT ROCm plugin, подключение к R9700
- `XlaBuffer` — обёртка над device buffer (загрузка/выгрузка данных CPU↔GPU)
- `XlaComputation` — скомпилированный HLO граф, готовый к запуску на GPU
- Хелперы: `matmul()`, `relu()`, `softmax()`, `gru_cell()` — строят HLO и компилируют
- Никакого Python — прямой вызов XLA C API из Rust

**actor_critic.rs** — Rust логика + XLA на GPU
- `ActorCritic` — actor (3 слоя) + critic (3 слоя), forward/backward через XLA на R9700
- `select_action()` — softmax с temperature из `CognitiveModulation.exploration_temperature`
- `batch_update()` — TD-error, policy gradient на GPU. **discount из `CognitiveModulation.temporal_discount`**

### Этап 5: HRL + Meta-Learner

**hrl.rs** — 150 строк
- `MetaController` — выбор скилла через XLA, temperature из CognitiveModulation
- `Skill` — policy + termination function, **impulse_control из CognitiveModulation**
- `HierarchicalRL` — связывает MetaController + Skills, `choose_action(&cog)`

**meta_learner.rs** — Rust + XLA на GPU
- `MetaLearner` — GRU + policy_head + value_head, GRU cell на GPU через XLA
- Вход: state + one_hot(last_action) + last_reward
- Рекуррентный контекст для мета-обучения

### Этап 6: World Model

**world_model.rs** — Rust + XLA на GPU
- `WorldModel` — VAE: encoder/decoder + GRU dynamics, все тензорные операции на R9700
- `encode()`, `decode()`, `predict_next_state()`

### Этап 7: Agent + Main

**agent.rs** — 280 строк
- `DecisionModule` enum — **enum dispatch** (Hierarchical / MetaLearning / ActorCriticOnly)
- `AgentConfig` — hidden_dim, num_skills, memory_size, use_tom, lr
- `PersonalityAgent` — единый struct со ВСЕМИ модулями
- `build_state_vector()` — STATE_DIM=10 из реальной нейрохимии + гомеостаза + позиции
- `choose_action()` — полный pipeline: CognitiveModulation → DecisionModule (enum dispatch) → PFC gating
- `process_step()` — среда → нейрохимия (threats, social) → динамика → память → обучение
- `learn()` — batch update с `temporal_discount` из CognitiveModulation

**main.rs** — 175 строк
- Инициализация XLA device (R9700 через PJRT ROCm)
- Создание агентов через `new_hierarchical()`, `new_meta_learning()`, `new_actor_critic()`
- Загрузка/сохранение чекпоинтов
- ToM: агенты наблюдают друг за другом
- Социальные встречи по manhattan_distance
- Мониторинг: speed, ETA, средние по популяции

---

## 5. Как нейрохимия управляет ВСЕМ (цепочка влияния)

```
NeurochemicalState (регионы, рецепторы)
        │
        ▼
compute_cognitive_modulation()
        │
        ├──► exploration_temperature ──► MetaController.select_skill()
        │                              ──► Skill.select_action()
        │                              ──► ActorCritic.select_action()
        │
        ├──► impulse_control ──────────► Skill.should_terminate()
        │
        ├──► learning_rate_multiplier ─► [для будущей адаптации LR]
        │
        ├──► temporal_discount ────────► ActorCritic.batch_update() discount
        │
        ├──► emotional_valence ────────► TheoryOfMind.mentalizing_capacity
        │
        ├──► risk_sensitivity ─────────► [для будущих решений]
        │
        └──► working_memory_capacity ──► TheoryOfMind.mentalizing_capacity

cortisol_level() ──────────────────────► PfcPlanner.gate_action() (вето)
serotonin_level() ─────────────────────► PfcPlanner.gate_action() (вето)
safety (homeostasis) ──────────────────► PfcPlanner.gate_action() (вето)
                                       ► PfcPlanner.derive_world_state()
                                       ► PfcPlanner.current_top_goal()
```

Это то, что было **полностью потеряно** в FINAL_INTEGRATION_*.rs, где нейрохимия заменена на 7 голых f32 без рецепторов, без регионов, без CognitiveModulation.

---

## 6. Что нужно для запуска

### Обязательно

1. **ROCm** — установлен (runtime 1.18), R9700 определяется как gfx1201
2. **XLA PJRT ROCm plugin** — скомпилированная `.so` библиотека XLA с поддержкой ROCm:
   ```bash
   # Собрать XLA из исходников с ROCm backend или скачать pre-built
   export XLA_PLUGIN_PATH=/path/to/pjrt_plugin_rocm.so
   export HSA_OVERRIDE_GFX_VERSION=12.0.1
   ```
3. **Cargo crate `xla`** — Rust FFI биндинги к XLA C API (в Cargo.toml)

### Архитектура GPU-слоя

```
Rust (логика, нейрохимия, GOAP, ToM)
  │
  ├── gpu.rs → XLA C API (FFI)
  │              │
  │              ├── PJRT ROCm plugin
  │              │
  │              └── R9700 (gfx1201, 30 GB VRAM, 64 CU)
  │
  └── Тензорные операции (matmul, softmax, GRU, backprop) → на GPU
      Логика агента, нейрохимия, планирование → на CPU
```

Никакого Python. Никакого PyTorch. Rust → XLA C API → ROCm → R9700.

### Сборка

```bash
cd /media/slava/B/AIAI
cargo build --release
```

---

## 7. Масштабирование

Текущая конфигурация (по умолчанию):
```
Агенты:      5
Шаги:        50,000
Мир:         20×20
Hidden dim:  128
Skills:      8
Memory:      100,000 эпизодов
```

Для extreme scale (поменять константы в main.rs и AgentConfig):
```
Агенты:      100
Шаги:        50,000,000
Мир:         200×200
Hidden dim:  512
Skills:      32
Memory:      2,000,000 эпизодов
Batch size:  2048
```

---

## 8. Файлы проекта — итоговая таблица

| Файл | Строк | Этап | Зависимости | Статус |
|------|-------|------|-------------|--------|
| `Cargo.toml` | 55 | — | — | Готов |
| `src/neurochemistry.rs` | 270 | 1 | std | Скомпилирован, протестирован |
| `src/homeostasis.rs` | 75 | 1 | — | Скомпилирован, протестирован |
| `src/memory.rs` | 65 | 1 | rand | Скомпилирован, протестирован |
| `src/pfc_planner.rs` | 215 | 2 | neurochemistry, homeostasis | Скомпилирован, протестирован |
| `src/world.rs` | 115 | 3 | std | Скомпилирован, протестирован |
| `src/theory_of_mind.rs` | 115 | 3 | neurochemistry, world | Скомпилирован, протестирован |
| `src/gpu.rs` | ~120 | 4 | xla (FFI) | Нужно написать |
| `src/actor_critic.rs` | 95 | 4 | gpu | Требует переписи на XLA |
| `src/hrl.rs` | 150 | 5 | neurochemistry, gpu | Создан, НЕ собирался |
| `src/meta_learner.rs` | 95 | 5 | world, gpu | Требует переписи на XLA |
| `src/world_model.rs` | 60 | 6 | gpu | Требует переписи на XLA |
| `src/agent.rs` | 280 | 7 | все модули | Создан, НЕ собирался |
| `src/main.rs` | 175 | 7 | все модули | Создан, НЕ собирался |
| **Итого** | **~1885** | | | |
