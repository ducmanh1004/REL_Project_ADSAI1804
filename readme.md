
**Q-Learning Guided Math Solver**

This project demonstrates a novel approach to solving complex, multi-step mathematics problems by combining the strategic decision-making of a classic Reinforcement Learning (RL) algorithm, Q-learning, with the advanced reasoning capabilities of a Large Language Model (LLM), Google's Gemini.

---

## Architecture Overview

The system employs a **"Learner-Judge"** architecture. A **QLearningAgent** acts as a **"manager,"** guiding a **"Learner"** LLM on the best high-level strategy to use for each step of a problem. A more powerful **"Judge"** LLM then evaluates the learner's work, providing a reward signal that teaches the Q-learning agent which strategies lead to correct solutions.

---

## What This Code Aims to Achieve

The primary goal is to create a system that can learn the optimal strategy for solving a class of mathematical problems. Instead of relying on a single, monolithic prompt for the entire problem, this code breaks the problem down into distinct steps. The Q-learning agent's task is to learn the best **"type" of action** to take at each step (e.g., **"apply a core rule"** vs. **"simplify the result").**

By doing this, the system can:

* **Improve Robustness:** If a mistake is made in one step, the system can identify it, provide feedback, and attempt the step again with a different strategy.
* **Enhance Explainability:** The chosen actions (**identify\_and\_setup**, **execute\_calculation**, etc.) provide a high-level, interpretable trace of the solution strategy.
* **Adapt and Learn:** Over time, the Q-learning agent's Q-table will store knowledge about which strategies are most effective for specific types of problems at specific stages of the solution process.

---

## Focus of the Training

A crucial point is what is actually being trained. **This code does not train or fine-tune the Gemini LLM.** The LLMs are used as powerful, pre-trained tools for reasoning and evaluation.

The training focus is entirely on the **QLearningAgent's Q-table.**

* The Q-table is a simple dictionary that maps **(state, action)** pairs to a numerical value (the **"Q-value"**), which represents the expected future reward of taking that action in that state.
* The agent's **"knowledge"** is stored in this table. The training process is about iteratively updating these Q-values so that the agent learns to select actions that maximize the total reward, which corresponds to correctly solving the math problem.

---

## Mapping the Q-Learning Algorithm to the Code

The **QLearningAgent** class is a direct implementation of the pseudocode provided in the image. Below is a breakdown of how each part of the algorithm corresponds to the Python code.

---

### The Algorithm

#### Mapping Breakdown

---

1. **Algorithm parameters: step size** $α ∈ (0,1]$, **small** ε > 0

   This corresponds to the initialization of the **QLearningAgent**.

   ```python
   class QLearningAgent:
       def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
           self.learning_rate: float = learning_rate  # This is α (alpha)
           self.discount_factor: float = discount_factor # This is γ (gamma)
           self.epsilon: float = epsilon            # This is ε (epsilon)
   ```

   *Use code with caution.*

2. **Initialize Q(s, a), for all** s ∈ S⁺, a ∈ A(s), arbitrarily except that Q(terminal, ·) = 0

   The Q-table is initialized as a **defaultdict**, which means values are created with a default of 0.0 as they are needed. This is a practical way of handling arbitrary initialization without pre-populating a massive table. The Q(terminal, ·) = 0 condition is handled by the fact that the **max\_next\_q** for a terminal state will be 0.

   ```python
   self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
   ```

   *Use code with caution.*

3. **Loop for each episode:**

   An episode in this context is the complete process of solving one math problem from start to finish. This loop is in the **main** function.

   ```python
   def main():
       # ...
       problems = create_hard_problems()
       for prob in problems: # <-- This is the loop for each episode
           solver.solve_problem(prob)
   ```

   *Use code with caution.*

4. **Loop for each step of episode:**

   A step is one of the 5 stages of solving the problem. This loop is inside the **solve\_problem** method. The state **S** is initialized at the beginning of this loop.

   ```python
   def solve_problem(self, problem: MathProblem):
       # ...
       for step in range(1, self.max_steps + 1): # <-- This is the loop for each step
           # Initialize S (State)
           state = self.agent.get_state(problem, step)
           # ...
   ```

   *Use code with caution.*

5. **State (S):** The state is a representation of the current situation. In this code, it's a string combining the problem's characteristics and the current step number.

   ```python
   def get_state(self, problem: MathProblem, step: int) -> str:
       return f"{problem.problem_type}_{problem.difficulty}_step{step}"
   # Example state: "calculus_hard_step2"
   ```

   *Use code with caution.*

6. **Choose A from S using policy derived from Q (e.g., ε-greedy)**

   The Action (A) is a high-level strategic choice (e.g., apply\_core\_rule). The **choose\_action** method implements the ε-greedy policy. With probability ε, it explores by choosing a random action; otherwise, it exploits by choosing the action with the highest Q-value for the current state.

   ```python
   def choose_action(self, state: str, step: int) -> str:
       # Exploration
       if random.random() < self.epsilon:
           return random.choice(available_actions)
       
       # Exploitation
       q_values_for_state = self.q_table[state]
       # ...
       return max(available_q_values, key=available_q_values.get)
   ```

   *Use code with caution.*

7. **Take action A, observe R, S'**

   This is a multi-part process in the **solve\_problem** method:

   * **Take Action A:** The chosen action is passed to the **"Learner"** LLM to generate the mathematical work for the step.

     ```python
     step_content = self.get_learner_response(problem, step, action, ...)
     ```

     *Use code with caution.*

   * **Observe Reward (R):** The **"Judge"** LLM evaluates the **step\_content** and returns a numerical reward.

     ```python
     is_correct, reward, judge_feedback = self.get_judge_evaluation(...)
     ```

     *Use code with caution.*

   * **Observe Next State (S'):** The next state is determined by incrementing the step counter.

     ```python
     next_step = step + 1
     next_state = self.agent.get_state(problem, next_step) if next_step <= self.max_steps else "terminal"
     ```

     *Use code with caution.*

8. **Q(S, A) ← Q(S, A) + α \[R + γ maxₐ Q(S', a) - Q(S, A)]**

   This is the core learning step. It is implemented exactly in the **update\_q\_value** method.

   ```python
   def update_q_value(self, state: str, action: str, reward: float, next_state: str, next_step: int):
       # Old Q-value: Q(S, A)
       current_q = self.q_table[state][action]

       # Find max_a Q(S', a)
       next_q_values = self.q_table[next_state]
       # ...
       max_next_q = max(relevant_next_q) if relevant_next_q else 0
       
       # Calculate the TD target: R + γ * max_a Q(S', a)
       td_target = reward + self.discount_factor * max_next_q
       
       # Calculate TD error: td_target - Q(S, A)
       td_error = td_target - current_q

       # The final update rule: Q(S, A) + α * TD_error
       new_q = current_q + self.learning_rate * td_error
       self.q_table[state][action] = new_q
   ```

   *Use code with caution.*

9. **S ← S'**

   This transition is handled implicitly by the for loop that iterates from **step = 1** to **step = 5**. When the loop finishes an iteration for the current step (S), it naturally moves to the next step (S') in the following iteration.
