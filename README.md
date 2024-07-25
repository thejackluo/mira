# Mira

Mira is an advanced AI chatbot designed to interact with users in a meaningful and context-aware manner. Mira incorporates various AI technologies, including tone analysis, context management, and memory integration, to provide a personalized and empathetic user experience.

## Table of Contents
- [Project Overview](#project-overview)
- [Current Features](#current-features)
- [Planned Features](#planned-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Mira aims to create a next-generation conversational AI that can understand and respond to users' emotional states, maintain context over long-term interactions, and provide personalized responses based on user profiles. The project is structured in iterative phases, each building on the previous to enhance Mira's capabilities.

## Current Features
### Iteration 1
- **Infrastructure Setup:**
  - ~~GitHub repository initialized.~~
  - ~~GPU setup for model training and inference.~~
- **Basic Chatting Pipeline:**
  - ~~Initial message handling pipeline.~~
  - ~~Capability to handle full message exchanges.~~
  - ~~Basic finetuning of models to assess initial performance.~~

### Iteration 2 (Ongoing)
- **Tone Determination:**
  - Small language model to analyze the tone of user messages and determine the AI's response tone.
- **Context Filling:**
  - Integration of a context-filling module.
  - Incorporation of Mira's profile for character traits and roles using prompt engineering.
- **Memory Space:**
  - Development of a memory pipeline for user profiles.
  - DynamoDB setup for long-term memory management.
- **Front-End Development:**
  - Basic online testing playground.
  - API and front-end integration for seamless user interaction.

### Iteration 3 (Planned)
- **Short-Term Memory:**
  - Implementation using text embeddings to maintain session context.
- **Emotional Computing:**
  - Exploration and integration of emotional computing for empathetic responses.
- **Advanced Context Processing:**
  - Regular inferences and deeper analysis of conversation context.
- **User Onboarding:**
  - Onboarding process to understand and store user information.

## Planned Features
- **Future Iterations:**
  - Advanced emotional computing.
  - Complex memory synthesis.
  - Nuanced conversational capabilities based on user feedback.

## Installation
To set up Mira locally, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/thejackluo/mira.git
    cd mira
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
   Create a `.env` file and add your configuration details.

4. **Run the application:**
    ```sh
    python main.py
    ```

## Usage
Mira can be interacted with through a terminal interface initially. As development progresses, a web-based interface will be available for a more user-friendly experience.
