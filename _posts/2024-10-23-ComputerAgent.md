---
title: Computer/Mobile Agent Papers Review
date: 2024-10-23 09:15
tags: Multi-Agent System
categories: Machine Learning
mathjax: true
---
# Developing a Computer use model -- Anthropic

[News website](https://www.anthropic.com/news/developing-computer-use)

- Building upon previous research in tool use and multimodality, researchers trained Claude to **understand and interact with computer screens**. This involved interpreting screenshots of the screen and using software tools to complete tasks.

- A crucial step was training Claude to **count pixels accurately**, allowing it to give precise mouse commands.

- Initially trained using **a few pieces of simple softwares** like a calculator and text editor, with no internet access to ensure safety. Despite this limited training, Claude **quickly learned to generalize, transforming user prompts into logical steps and actions, and even self-correcting when encountering challenges**. 

- Claude currently represents the state-of-the-art in models using computers like humans. While its performance on the OSWorld evaluation is far better than other models, it still **falls short of human-level skill. (14.9% / 70% - 75%)**

# Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale

[Arxiv](https://arxiv.org/abs/2409.08264)

[Repo](https://github.com/microsoft/WindowsAgentArena)

## Task

Formalize the agent bahavior with state space $S$, observation space $O$, action space $A$, transition function $T: S \times A \rightarrow S$ and reward function $R: S \times A \rightarrow \mathbb{R}$.

Given that current observation $o_t \in O$, and agent execute action $a_t \in A$ returns the new observation $o_{t+1} \in O$ and new state $s_{t+1} \in S$.

The reward function $R$ will return a value $\in [0, 1]$ to indicate the degree of completion of the task.

### Observation 

- Foreground and background window titles by `pygetwindow`

- Clipboard content by `pyperclip` for texts and store an VLM-generated description for images

- Acessbility tree by `pywinauto` but do not feed the UIA tree directly. Instead some agents configurations parse the tree to extract contents

- Screenshot of previous screen

- Screenshot of current screen

### Action

Two action spaces:

- free-form `pyautogui/python` code execution

- function wrappers through the `Computer` class

## Baseline method

By Chain-of-thought, use current stage of the computer and model its own past actions to decide the most appropriate next action.

- UIA tree parsing: extracts the visible elements from the Windows UI Automation tree

- DOM tree parsing: extracts the visible elements from the DOM tree (browser only)

- OCR: proprietary and open models (Tesseract)

- Icon and image detection: proprietary and open models (Grounding DINO)
OmniParser: proprietary model that detects detects text, icons, and images and provides icon captioning

# OSWORLD: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

[Arxiv](https://arxiv.org/abs/2404.07972)

[Repo](https://github.com/xlang-ai/OSWorld)

Similar work to WAA, but in Linux system.

# Mobile App Tasks with Iterative Feedback (MoTIF): Addressing Task Feasibility in Interactive Visual Environments

[Arxiv](https://arxiv.org/abs/2104.08560)

[Repo](https://github.com/aburns4/MoTIF)

A dataset of mobile app tasks with iterative feedback.

# Mind2Web: Towards a Generalist Agent for the Web

[Arxiv](https://arxiv.org/abs/2306.06070)

[Repo](https://github.com/OSU-NLP-Group/Mind2Web)

A dataset with natural language tasks and manually annotated action sequences for developing and evaluating generalist agents for the web.

- Diverse converage of domains, websites, and tasks

- Use of real-world websites

- A broad spectrum of user interaction patterns

## Task

Avoid low-level, step-by-step instructions, aiming to foster the development of agents that can comprehend and carry out tasks in a more autonomous fashion, rather than merely following prescriptive directives.

- Action sequence: $(Target, Element, Action)$ as a pair

- Webpage snapshots: `HTML` code, `DOM` tree and `HAR` files

## Method

### Candidate Generation with Small LMs

1.	**Extract DOM nodes:** The system extracts multiple DOM nodes from the HTML structure of the webpage (i.e., various HTML elements like `<div>`, `<button>`, etc.), with each node treated as a candidate object.

2.	**Generate task query:** The system combines the task description and the previous actions to form a task query.

3.	**Rank the DOM nodes:** Each DOM node is represented based on its tag, text content, attribute values, and the relevant information from its parent and child nodes. It is then paired with the task query and matched using a language model, generating a score.

4.	**Select the most relevant DOM nodes:** The system ranks the DOM nodes based on the generated scores and selects the top-k nodes with the highest scores as the candidates for the next action (e.g., clicking, input, etc.).

### Action Prediction with LLMs

1.	**Webpage snapshot:** After identifying the top-k candidate DOM elements, the system prunes the webpage snapshot, keeping only the selected candidates and their neighboring elements. These snippets are used as input to a large language model (LLM).

2.	**Reformulate task as multi-choice QA:** Instead of generating the target DOM element, the task of element selection is converted into a multiple-choice question. The LLM is trained to select the correct element from a list of options.

3.	**Comparison with direct generation:** A baseline method that directly generates the target DOM element based on the pruned webpage snippet is also included for comparison. In both methods, the LLM generates the required operation, and any additional values necessary for the action.

4.	**Training and inference process:** During training, the model is fine-tuned with ground-truth actions using a left-to-right language modeling objective. At inference time, the top-k candidates are grouped into clusters of five options (including a “None” option). If more than one option is selected, the process repeats with new groups formed from the selected elements, until a single element is chosen or none are selected.

# Mapping Natural Language Instructions to Mobile UI Action Sequences

[Arxiv](https://arxiv.org/abs/2005.03776)

[Repo](https://github.com/google-research/google-research/tree/master/seq2act)

Three datasets for natural language instructions to mobile user interface actions.

## Problem formulation

Given an instruction of a multiple-step task $I = (t_1, t_2, ..., t_n)$, where $t_i$ is the $i$ th token in instruction $I$.

<span style="text-decoration: underline;">**Goal:**</span> Generate a sequence of automatically executable actions $A = (a_1, a_2, ..., a_m)$ over a sequence of user interface screen $S$ with initial screen $S_1$ and transition function $s_j = \tau(s_{j-1}, a_{j-1})$: 

$$p(A | s_1, \tau, I) = \prod_{j=1}^{m} p(a_j | a_{<j}, s_{j-1}, \tau, I)$$

# AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents

[Arxiv](http://export.arxiv.org/abs/2405.14573)

[Repo](https://github.com/google-research/android_world)

