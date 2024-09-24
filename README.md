# RecLMM

## 🚀 About The Project

RecLLM is an effort to bring interactive, LLM-based recommender systems at Instacart. It is a spin-off from the RecAI project from Microsoft here: https://github.com/microsoft/RecAI/tree/main/InteRecAgent

## 🛠️ Getting Started

### Prerequisites

See requirements.txt

### Installation

1. Clone the repo
   ```bash
   git clone https://github.com/PantelisInsta/RecLLM.git
   ```
2. Create compatible virtual environment
  ```bash
  cd RecLLM
  pyenv local 3.9.13
  python -m venv RecLLM
  ```
3. Source environment and install dependencies
  ```bash
  source RecLLM/bin/activate
  pip install -r requirements.txt
  ```

For more instructions, check the original README file here: https://github.com/microsoft/RecAI/blob/main/InteRecAgent/README.md

## 🔬 Running Use-cases

### Use-case 1: Affordability Basket

#### With reflection
```bash
python app.py --affordability_basket --enable_reflection 1
```

#### Without reflection
```bash
python app.py --affordability_basket --enable_reflection 0
```

### LLM Ranker
```bash
python app.py --LLM_ranker
```

## 📊 Running Evaluations

To evaluate the LLM ranker, use the following command:

```bash
python eval/eval_LLM_ranker.py
```

This will run the evaluation script for the LLM ranker and save performance metrics.