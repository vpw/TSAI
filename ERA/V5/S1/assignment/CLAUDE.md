# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is the S1 (Session 1) assignment submission for the ERA V5 course (The School of AI). It is a single self-contained `index.html` file that demonstrates 4 deep learning concepts interactively in the browser, using in-browser training (no backend, no build step).

Read `S1-assignment.md` for the exact spec/claims each exercise must prove, and `AGENTS.md` for the submission requirements (single HTML+JS deliverable, deployed to Netlify).

## Running / developing

There is no build system, package manager, or test suite — this is a static file.

- To develop: just open `index.html` directly in a browser (or serve it with any static file server, e.g. `python3 -m http.server`).
- There is no lint/build/test command to run. Verify changes by opening the file in a browser and exercising each tab's buttons.
- Deployment target is Netlify: the single `index.html` is uploaded/dragged to a Netlify site for the final submission.

## Architecture

Everything lives in one file, `index.html`, organized as:

- A `<head>` that pulls in two ML libraries from CDN: **TensorFlow.js 4.10.0** and **Brain.js**.
- A `<body>` with 4 tab buttons (`showTab('s1-1' | 's1-2' | 's1-3' | 's1-4')`) that toggle visibility of 4 corresponding `<div>` sections, each with its own `<canvas>` elements for plots.
- One big `<script>` block containing all logic for all 4 tabs (no modules, no imports — everything is a global function).

Each of the 4 tabs is an independent demo with its own data generation, model(s), training loop, and canvas-drawing code. They don't share state except for the `showTab` mechanism. Key thing to know when editing: **S1-1/S1-2/S1-3 use TensorFlow.js**, while **S1-4 uses Brain.js** instead (a different library, different API shape — `trainBrainJS`/`evalBrainJS` wrap brain.js's neural network rather than a tf.js model).

### Per-tab breakdown

- **S1-1 (Activations)** — `generateRingData`, `createLinearModel`, `createReLUModel`, `plotDecisionBoundary`, `trainS1_1`. Proves a single linear+sigmoid model can't separate concentric rings (~55% acc, straight boundary) while adding one ReLU hidden layer can (~99%, curved boundary).
- **S1-2 (Depth without nonlinearity)** — `generateDataS1_2`, `create1LayerModel`, `create5LinearModel`, `create5ReLUModel`, `trainS1_2`, `showMatrixProof`. Proves 5 stacked linear layers collapse to one linear map (identical accuracy/boundary to 1 layer) until ReLU is inserted between them. The "bonus" matrix-multiplication proof multiplies the 5 weight matrices numerically to show the product is a single matrix.
- **S1-3 (Embeddings)** — `generateGrammarData`, `trainS1_3`, `computePCA`/`projectTo2D`, `plotEmbeddings`, `computeNearestNeighbors`. Trains a tiny embedding→softmax next-token model on a synthetic grammar (animals/fruits/verbs categories) and shows the learned embeddings cluster by category even though category was never a training signal.
- **S1-4 (Memorization vs generalization)** — `generateData`, `trainBrainJS`, `evalBrainJS`, `runS1_4`, `visualizeS1_4Data`, `drawLossHistoryChart`, `drawLossChart`, `drawGapChart`. Trains an over-parameterized brain.js network at 3 dataset sizes (20/200/2000) and plots the shrinking train/test loss gap as data grows. Uses per-size-tuned epochs/learning-rate/momentum since brain.js's optimizer behaves differently from tf.js.

### Canvas plotting conventions

All visualizations are hand-drawn on `<canvas>` via 2D context (no charting library). Each demo has its own `coordToCanvas`-style normalization between data space and pixel space — when adding a new plot, follow the existing pattern in the relevant tab's plotting function rather than introducing a new charting approach.
