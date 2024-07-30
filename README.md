# DA-RNN Implementation

## Overview

This repository contains an implementation of the **DA-RNN (Dual Attention Recurrent Neural Network)**. DA-RNN is a sophisticated model designed for time series forecasting. It leverages both temporal and contextual attention mechanisms to enhance predictive accuracy.

## Key Features

- **Temporal Attention Mechanism**: Focuses on the significance of different time steps in the sequence.
- **Contextual Attention Mechanism**: Captures long-term dependencies and contextual relationships.
- **Improved Predictive Accuracy**: Better captures complex patterns in sequential data.
## Prerequisites

- Python 3.x

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/valentin-fngr/Dual_Stage_attention_RNN.git
   cd Dual_Stage_attention_RNN

2. Install the required packages: 
    ```bash 
    pip install -r requirements.txt

## Usage

The train.py script accepts the following command-line arguments:

    --config: Path to the configuration file.
    --checkpoint: Directory to save checkpoints.
    --seed: Random seed for reproducibility (default: 0).
