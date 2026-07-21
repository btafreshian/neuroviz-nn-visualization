# NeuroViz

NeuroViz is an interactive neural-network playground for learning what happens during training. Instead of treating a model as a black box, it lets you click a data point, follow its forward pass through the network, and watch the weights, activations, prediction confidence, loss, and decision boundary change together.

Everything runs in the browser. There is no backend, account, database, or external model service.

## What you can explore

- Train a real `2 → 8 → 6 → 1` feed-forward classifier with seeded initialization and deterministic datasets.
- Switch between XOR, two moons, concentric circles, two spirals, and Gaussian blobs.
- Adjust the loss function, learning rate, epoch count, sample count, noise, seed, and train/test split.
- Run continuously, pause, resume, or advance exactly one epoch at a time.
- Click a sample to inspect its coordinates, true class, predicted class, probability, and individual loss.
- Trace neuron activations, pre-activations, biases, edge weights, updates, and per-edge contributions.
- Compare train and validation loss, accuracy, confidence, gradient size, update size, and weight norm.
- Scrub through completed epochs without changing the live model or optimizer state.
- Reset the live model to its zero-weight state while keeping the current run available for replay.

The decision boundary is drawn from the model's actual probabilities. Pale areas are uncertain, saturated areas are confident, and red rings identify misclassified samples.

## Run it locally

You will need Node.js 20.19 or newer and pnpm 11.9.0.

```bash
corepack enable
corepack prepare pnpm@11.9.0 --activate
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

For a production-style local run:

```bash
pnpm build
pnpm start
```

No environment variables are required.

## Quality checks

Run the complete non-browser validation suite with:

```bash
pnpm check
```

That command runs TypeScript, ESLint, unit tests, and a production build. Individual commands are also available:

```bash
pnpm typecheck
pnpm lint
pnpm test:run
pnpm build
```

The browser regression tests cover training, sample tracing, replay, reset behavior, refresh behavior, and responsive overflow:

```bash
pnpm exec playwright install chromium
pnpm test:e2e
```

## Project layout

| Path | Responsibility |
| --- | --- |
| `app/` | Next.js entry point, metadata, icon, and global styles |
| `components/graph-canvas/` | Interactive network diagram and neuron/edge inspection |
| `components/training/` | Controls, diagnostics, charts, decision boundary, and epoch replay |
| `components/inspector/` | Dataset, network, and hyperparameter panels |
| `lib/context/network-context.tsx` | Live training session, UI state, snapshots, and replay isolation |
| `lib/nn/simple-trainer.ts` | Forward pass, backpropagation, optimizers, evaluation, and trace calculations |
| `lib/datasets/` | Deterministic synthetic dataset generation and stratified splitting |
| `tests/e2e/` | Browser-level regression tests |

## How training and replay work

The live training session owns the model parameters and optimizer buffers. Each completed epoch records telemetry plus an immutable snapshot of weights, biases, and weight updates. Replay reads those snapshots through a separate visualization state, so moving the epoch slider cannot mutate the live model.

The initial screen is a genuine zero-weight model. Starting training introduces the seeded parameters with a short transition before epoch 1. Reset returns the live visualization and metrics to the zero state but intentionally keeps the current run's snapshots. Reloading the page clears all in-memory history.

## Current scope

NeuroViz currently focuses on binary classification with two input features and a fixed network architecture. Training history lives only in memory, so it is not persisted between page loads. Those constraints keep the playground focused on explaining the mechanics of a forward pass and the evolution of a decision boundary.
