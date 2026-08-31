# ERA V5 S1 - Implementation Summary

## Overview
Single HTML file (`index.html`) with 4 interactive tabbed demonstrations of deep learning fundamentals using TensorFlow.js.

## File Structure
```
/u/Vardhan/Courses/TSAI/ERA/V5/S1/assignment/
├── index.html          # Main implementation (~1400 lines)
├── AGENTS.md           # Assignment instructions
├── S1-assignment.md    # Detailed exercise descriptions
└── IMPLEMENTATION_SUMMARY.md  # This file
```

---

## S1-1: Activations Exist For A Reason ✓

### Concept
Demonstrates that linear models can only draw straight boundaries, while ReLU enables curved boundaries to separate non-linearly separable data.

### Data
- 300 noisy 2D points as two concentric rings
- Inner ring = class 0 (red), Outer ring = class 1 (teal)

### Models
1. **Linear Model**: Single dense layer (2→1) + sigmoid → draws straight line
2. **ReLU Model**: Input → Dense(64) → ReLU → Dense(1) → sigmoid → wraps ring

### UI Elements
- "🎲 Generate New Data" button → plots data points
- "▶ Train Both Models" button → trains and shows decision boundaries
- Two side-by-side canvases (400x400) showing:
  - Linear: blue/orange regions, stuck at ~55%
  - ReLU: red/green gradient, ~99% accuracy

### Key Functions
- `generateRingData(nPoints, noise)` - generates concentric ring data
- `createLinearModel()` - 1-layer linear model
- `createReLUModel()` - 1 hidden layer with ReLU
- `plotDecisionBoundary()` - visualizes decision regions
- `plotDataOnly()` - shows just the data points

---

## S1-2: Depth Without Nonlinearity Is A Lie ✓

### Concept
5 stacked linear layers collapse to a single linear map (W5·W4·W3·W2·W1 = W_final). Without ReLU, depth adds nothing!

### Data
- Same ring data as S1-1

### Models
1. **1 Linear Layer**: Single dense(1) + sigmoid
2. **5 Linear Layers**: Dense(32) → Linear × 4 → Dense(1) → sigmoid (NO ReLU)
3. **5 Layers + ReLU**: Same architecture but with ReLU after each hidden layer

### UI Elements
- "🎲 Generate New Data" button
- "▶ Train All 3 Models" button
- 3 smaller canvases (300x300) side-by-side
- Bonus: Matrix multiplication proof section

### Key Functions
- `create1LayerModel()` - baseline
- `create5LinearModel()` - 5 linear layers (no activation)
- `create5ReLUModel()` - 5 layers with ReLU
- `showMatrixProof()` - displays weight matrix info

### Expected Result
- 1-Layer and 5-Linear have **identical accuracy** (~55%)
- Both draw the **same straight line** (mathematical proof!)
- 5-ReLU achieves ~99% with curved boundary

---

## S1-3: Embeddings Learn Similarity From Next-Token ✓

### Concept
Trained only to predict next tokens, embeddings cluster related items together even though similarity was never supplied.

### Vocabulary
| Category | Tokens |
|----------|--------|
| Animals (red) | cat, dog, cow |
| Fruits (yellow) | apple, mango |
| Verbs (teal) | eat, chase, see |

### Grammar Templates
1. `[animal] [verb]` → "cat eat"
2. `[animal] [verb] [animal]` → "dog chase cat"
3. `[fruit] is tasty` → "apple is tasty"
4. `[verb] [animal]` → "see dog"

### Model
- Embedding(8 tokens → 16 dim) → Flatten → Dense(8, softmax)
- Trained to predict next token from current token

### UI Elements
- "▶ Train Embeddings" button
- Canvas showing 2D projection of learned embeddings
- Nearest neighbors display for each token

### Key Functions
- `generateGrammarData()` - creates training pairs from grammar
- `trainS1_3()` - trains embedding model
- `projectTo2D()` - simple projection for visualization
- `plotEmbeddings()` - draws embedding scatter plot
- `computeNearestNeighbors()` - finds similar tokens

### Expected Result
- Animals cluster together
- Fruits cluster together
- Verbs cluster together
- Nearest neighbors show same-category tokens

---

## S1-4: Memorization vs Generalization (NOT YET IMPLEMENTED)

### Planned Implementation
- Train over-parameterized network on dataset sizes: 20, 200, 2000
- Show train/test loss gap
- Demonstrate that more data = better generalization

---

## Technical Notes

### Global Variables
- `currentData` - stores S1-1 ring data (xs, ys tensors)
- `currentDataS1_2` - stores S1-2 ring data
- `trainedModelsS1_2` - stores trained models for matrix proof

### Color Scheme
- Background: #1a1a2e → #16213e gradient
- Primary: #4a90d9 (blue)
- Success: #51cf66 (green)
- Warning: #ff6b6b (red)
- Class 0: #ff6b6b (red)
- Class 1: #4ecdc4 (teal)

### Libraries
- TensorFlow.js 4.10.0 (CDN)

### Browser Requirements
- Modern browser with WebGL support (for TensorFlow.js)

---

## Running Locally
Simply open `index.html` in a web browser. No server required.

## Future Improvements
1. Complete S1-4 implementation
2. Add export/reset functionality
3. Add more detailed explanations
4. Mobile responsiveness improvements
5. Add t-SNE or UMAP for better embedding visualization in S1-3
