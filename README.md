# 🤖 Comprehensive LLM Agent Research Collection

<div align="center">

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
[![commit](https://img.shields.io/github/last-commit/luo-junyu/Awesome-Agent-Papers?color=blue)](https://github.com/luo-junyu/Awesome-Agent-Papers/commits/main)
[![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://github.com/luo-junyu/Awesome-Agent-Papers/pulls)

</div>

<p align="center">
  <img src="./figs/fig-overview-agent-survey.png" width="90%" alt="LLM Agent Research Overview">
</p>

## 🌟 Overview

This repository contains a **comprehensive collection** of research papers on Large Language Model (LLM) agents. We organize papers across key categories including agent construction, collaboration mechanisms, evolution, tools, security, benchmarks, and applications.

Our taxonomy provides a structured framework for understanding the rapidly evolving field of LLM agents, from architectural foundations to practical implementations. The repository bridges fragmented research threads by highlighting connections between agent design principles and emergent behaviors.

📄 **[Read our survey paper here](https://arxiv.org/abs/2503.21460)**


Our survey covers the rapidly evolving field of LLM agents, with a significant increase in research publications since 2023.


## 📑 Table of Contents

- [🌟 Overview](#-overview)
- [📊 Statistics & Trends](#-statistics--trends)
- [🔍 Key Categories](#-key-categories)
- [📚 Resource List](#-resource-list)
  - [Agent Collaboration](#agent-collaboration)
  - [Agent Construction](#agent-construction)
  - [Agent Evolution](#agent-evolution)
  - [Applications](#applications)
  - [Datasets & Benchmarks](#datasets--benchmarks)
  - [Ethics](#ethics)
  - [Security](#security)
  - [Survey](#survey)
  - [Tools](#tools)
- [🤝 Contributing](#-contributing)

## 🔍 Key Categories

- **🏗️ Agent Construction**: Methodologies and architectures for building LLM agents
- **👥 Agent Collaboration**: Frameworks for multi-agent interaction and cooperation
- **🌱 Agent Evolution**: Self-improvement and learning capabilities of agents
- **🔧 Tools**: Integration of external tools and APIs with LLM agents
- **🛡️ Security**: Security concerns and protections for LLM agent systems
- **📊 Benchmarks**: Evaluation frameworks and datasets for testing agent capabilities
- **💡 Applications**: Real-world implementations and use cases


## 📚 Resource List

### Agent Collaboration

- **[Foam-Agent: Towards Automated Intelligent CFD Workflows](https://arxiv.org/abs/2505.04997)** (*2025*)
  > The paper presents Foam - Agent, a multi - agent framework automating CFD workflows from natural language. It features unique retrieval, file - generation and error - correction systems, lowering expertise barriers.

- **[Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)** (*2025*)
  > Presents MAST, a taxonomy for MAS failures. Develops an LLM-as-a-Judge pipeline, and opensources data to guide MAS development.

- **[Linear formation control of multi-agent systems](https://www.sciencedirect.com/science/article/pii/S0005109824004291)** (*2025*)
  > A new distributed leader–follower control architecture (linear formation control) is proposed for formation variations, with new concepts and estimation methods.

- **[MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents](https://arxiv.org/abs/2503.01935)** (*2025*)
  > Introduces MultiAgentBench to evaluate LLM - based multi - agent systems. Assesses collaboration and competition, evaluates protocols and strategies, code & data open - sourced.

- **[A Survey of AI Agent Protocols](https://arxiv.org/abs/2504.16736)** (*2025*)
  > Paper analyzes existing LLM agent protocols, proposes a classification, explores future directions for next - gen protocols.

- **[C^2: Scalable Auto-Feedback for LLM-based Chart Generation](https://aclanthology.org/2025.naacl-long.232/)** (*2025*)
  > The paper introduces C2, a framework with an auto - feedback provider and a reference - free dataset, eliminating human curation, open - sourced at chartsquared.github.io.

- **[AgentRxiv: Towards Collaborative Autonomous Research](https://arxiv.org/abs/2503.18102)** (*2025*)
  > Introduces AgentRxiv, a framework enabling LLM agent labs to share research on a preprint server for collaboration, aiding future AI design with humans.

- **[Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains](https://arxiv.org/abs/2501.05707)** (*2025*)
  > Proposes multiagent finetuning for language models. Specializes models via multiagent - generated data, preserving diverse reasoning chains for better self - improvement.

- **[From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](https://arxiv.org/abs/2506.08292)** (*2025*)
  > Proposes ECON, a hierarchical RL paradigm recasting multi - LLM coordination as a BNE game, with tighter regret bound and scalability.

- **[Chain of Agents: Large language models collaborating on long-context tasks](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/)** (*2025*)
  > Proposes Chain-of-Agents, a training-free, task-agnostic framework using LLM collaboration for long-context tasks, outperforming RAG and long-context LLMs.

- **[Adaptive Collaboration Strategy for LLMs in
Medical Decision Making](https://arxiv.org/abs/2404.15155)** (*2024*)
  > Proposes Medical Decisionmaking Agents (MDAgents) to assign LLM collaboration structures, adapting to task complexity and exploring group consensus.

- **[ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs](https://arxiv.org/abs/2309.13007)** (*2024*)
  > Proposes ReConcile, a multi - model multi - agent framework like a round - table conference, enhancing LLM collaborative reasoning via discussion and voting.

- **[MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)** (*2024*)
  > Introduces MetaGPT, a meta - programming framework integrating human workflows into LLM - based multi - agent systems, improving task breakdown and error reduction.

- **[Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782)** (*2024*)
  > The paper explores if weaker models can assess stronger ones via debate. It shows debate helps non - experts and optimising debaters aids truth - finding without ground truth.

- **[Roco: Dialectic multi-robot collaboration with large language
models](https://arxiv.org/abs/2307.04738)** (*2024*)
  > Proposes using pre - trained LLMs for multi - robot high - level comm. and low - level path planning, with in - context improvement, and introduces RoCoBench.

- **[AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning](https://arxiv.org/abs/2401.05268)** (*2024*)
  > AutoAct is an automatic QA agent learning framework. It synthesizes trajectories without external help and uses labor - division for task completion.

- **[Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding](https://arxiv.org/abs/2401.12954)** (*2024*)
  > Introduces meta - prompting, a task - agnostic scaffolding to turn one LM into a multi - role system, integrates external tools, enhancing task performance.

- **[Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://arxiv.org/abs/2305.19118)** (*2024*)
  > The paper proposes a Multi - Agent Debate framework to solve the DoT problem in LLMs, encouraging divergent thinking for complex tasks.

- **[AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors](https://openreview.net/forum?id=EHg5GDnyq1)** (*2024*)
  > The paper proposes AgentVerse, a multi - agent framework inspired by human group dynamics, facilitating collaboration and revealing emergent behaviors in agents.

- **[Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration](https://arxiv.org/abs/2310.02170)** (*2024*)
  > Proposes DyLAN, a framework for LLM - powered agent collaboration. It uses two - stage paradigm with agent selection and dynamic collaboration for various tasks.

- **[ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)** (*2024*)
  > This paper presents ChatDev, a chat - powered framework. Specialized LLM agents collaborate via unified linguistic comm., bridging phases for autonomous task - solving.

- **[ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate](https://openreview.net/forum?id=FQepisCUWu)** (*2024*)
  > The paper presents ChatEval, a multi - agent referee team, leveraging multi - agent debate for text evaluation, offering a human - mimicking process.

- **[A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration](https://openreview.net/forum?id=XII0Wp1XA9#discussion)** (*2024*)
  > This paper proposes DyLAN, a framework for LLM - powered agent collaboration. It selects agents dynamically and uses a two - stage paradigm for task - solving.

- **[AgentCoord: Visually Exploring Coordination Strategy for LLM-based Multi-Agent Collaboration](https://arxiv.org/pdf/2404.11943)** (*2024*)
  > Presents a visual exploration framework for multi - agent coordination strategy design, converts goals to strategies, allowing user intervention.

- **[TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138)** (*2024*)
  > This paper proposes a novel stock trading framework with LLM - powered agents in specialized roles, simulating real - world collaboration to improve trading performance.

- **[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)** (*2023*)
  > AutoGen is an open - source framework enabling LLM app building with multi - agent conversation, customizable agents, and flexible interaction definition.

- **[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)** (*2023*)
  > A multi - agent debate approach is presented to improve LLMs' reasoning and factuality, applicable to black - box models with a unified procedure.

- **[Autonomous chemical research with large
language models](https://www.nature.com/articles/s41586-023-06792-0)** (*2023*)
  > The paper introduces Coscientist, an AI system driven by GPT - 4. It integrates tools, shows potential in research, and demonstrates AI's versatility and efficacy.

### Agent Construction

- **[Planning with Multi-Constraints via Collaborative Language Agents](https://aclanthology.org/2025.coling-main.672/)** (*2025*)
  > This paper proposes PMC, a zero - shot method for LLM - based multi - agent systems. It simplifies complex, constraint - heavy task planning via task decomposition.

- **[Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b631da756d1573c24c9ba9c702fde5a9-Abstract-Datasets_and_Benchmarks_Track.html)** (*2025*)
  > The paper proposes an Embodied Agent Interface to unify tasks, modules, and metrics, comprehensively assessing LLMs for embodied decision making.

- **[SPeCtrum: A Grounded Framework for Multidimensional Identity Representation in LLM-Based Agent](https://arxiv.org/abs/2502.08599)** (*2025*)
  > Introduces SPeCtrum, a framework integrating S, P, C for LLM agent personas. Enhances identity realism, enabling personalized AI interactions.

- **[Adaptive Thinking via Mode Policy Optimization for Social Language Agents](https://arxiv.org/pdf/2505.02156)** (*2025*)
  > Proposes Adaptive Mode Learning (AML) framework and AMPO algorithm, offering multi - granular modes, context - aware switching, and token - efficient reasoning.

- **[On Architecture of LLM agents](http://www.injoit.ru/index.php/j1/article/view/2057)** (*2025*)
  > The paper discusses LLM agent architecture. Agents are a key area in AI, acting like mashups and robots, and frameworks can simplify their creation.

- **[Unified Mind Model: Reimagining Autonomous Agents in the LLM Era](https://arxiv.org/abs/2503.03459)** (*2025*)
  > This paper proposes the Unified Mind Model (UMM) for human - level agents. It also develops MindOS to create task - specific agents without programming.

- **[ATLaS: Agent Tuning via Learning Critical Steps](https://arxiv.org/abs/2503.02197)** (*2025*)
  > Proposes ATLaS to identify critical steps in expert trajectories for LLM agent tuning, reducing cost and enhancing generalization.

- **[Cognitive AI Memory: A Framework for More Human-like Memory in LLMs](https://arxiv.org/abs/2505.13044)** (*2025*)
  > The paper proposes CAIM framework inspired by cognitive AI for LLMs, with three modules, enhancing long - term human - AI interaction by holistic memory modeling.

- **[Adaptive Graph Pruning: A Task-Adaptive Multi-Agent Collaboration Framework](https://arxiv.org/abs/2506.02951)** (*2025*)
  > Proposes Adaptive Graph Pruning (AGP), a task - adaptive multi - agent framework jointly optimizing agent quantity and communication topology via a two - stage strategy.

- **[Agents of Change: Self-Evolving LLM Agents for Strategic Planning](https://arxiv.org/abs/2506.04651)** (*2025*)
  > Puts LLM agents in strategic - challenging environments, uses Catan game for benchmarking, and proposes a multi - agent architecture for self - improvement.

- **[Reinforcing Large Language Model Reasoning through Multi-Agent Reflection](https://arxiv.org/abs/2506.08379)** (*2025*)
  > The paper models multi - turn refinement as an MDP and introduces DPSDP, a RL algorithm for iterative answer refinement, showing theoretical and empirical benefits.

- **[Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/abs/2508.19828)** (*2025*)
  > Proposes Memory - R1, an RL framework with two agents for LLMs to actively manage and utilize external memory, offering insights into RL - enabled behavior.

- **[BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens](https://arxiv.org/abs/2508.17196)** (*2025*)
  > Introduces BudgetThinker, a framework for budget - aware LLM reasoning. Inserts control tokens and uses a two - stage training pipeline for efficient, controllable reasoning.

- **[A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)** (*2025*)
  > Paper proposes an agentic memory system for LLMs, organizing memories like Zettelkasten, enabling dynamic updates and more adaptive memory management.

- **[AutoAgents: A Framework for Automatic Agent Generation](https://arxiv.org/abs/2309.17288)** (*2024*)
  > Introduces AutoAgents, a framework generating and coordinating specialized agents per task. Incorporates an observer. Offers new complex - task - tackling perspectives.

- **[MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)** (*2024*)
  > MetaGPT is a meta - programming framework integrating human workflows into LLM - based multi - agent collaborations, streamlining workflows and reducing errors.

- **[Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)** (*2024*)
  > Proposes CoALA, a framework for language agents with modular memory, action space, and decision - making, organizing work and guiding future development.

- **[Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030)** (*2024*)
  > This work proposes CodeAct using executable Python code for LLM agents, unifying action space, and builds an open - source agent with a tuned dataset.

- **[ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)** (*2024*)
  > The paper introduces ChatDev, an LLM - powered framework enabling agents to collaborate via language for software design, coding, and testing, unifying phases.

- **[Editable Scene Simulation for Autonomous Driving via Collaborative LLM-Agents](https://openaccess.thecvf.com/content/CVPR2024/papers/Wei_Editable_Scene_Simulation_for_Autonomous_Driving_via_Collaborative_LLM-Agents_CVPR_2024_paper.pdf)** (*2024*)
  > This paper presents ChatSim, enabling editable 3D driving scene sims via NLP. It uses LLM - agent, new neural radiance field and lighting estimation methods.

- **[A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration](https://arxiv.org/abs/2310.02170)** (*2024*)
  > A framework named DyLAN is proposed for LLM - powered agent collaboration. It has a two - stage paradigm with dynamic agent selection and communication for tasks.

- **[More Agents Is All You Need](https://arxiv.org/abs/2402.05120)** (*2024*)
  > The paper proposes Agent Forest, a sampling - and - voting method. It's orthogonal to existing ones, enhancing LLMs with performance correlated to task difficulty.

- **[Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/abs/2405.02957)** (*2024*)
  > Presents Agent Hospital, a hospital simulacrum with LLM - powered agents. Doctor agents evolve without manual labeling, and methods benefit broader apps.

- **[Empowering biomedical discovery with AI agents](https://www.cell.com/cell/fulltext/S0092-8674(24)01070-5?&target=_blank)** (*2024*)
  > Paper proposes “AI scientists” as collaborative agents integrating AI and bio - tools. They combine human and AI abilities and impact multiple bio - areas.

- **[SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models](https://ieeexplore.ieee.org/abstract/document/10802322)** (*2024*)
  > Proposes SMART-LLM, an LLM - based framework for multi - robot task planning. Creates a benchmark dataset and offers resources on https://sites.google.com/view/smart-llm/.

- **[Perceive, Reflect, and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions](http://arxiv.org/abs/2408.04168)** (*2024*)
  > The paper presents a novel LLM agent workflow with perception, reflection, and planning for goal - directed city navigation, avoiding baseline drawbacks.

- **[Enhancing the General Agent Capabilities of Low-Parameter LLMs through Tuning and Multi-Branch Reasoning](https://arxiv.org/abs/2403.19962)** (*2024*)
  > Proposes constructing agent - specific data with GPT - 4 and fine - tuning small - parameter LLMs. Multi - path reasoning and task decomposition improve agent performance.

- **[PlanCritic: Formal Planning with Human Feedback](https://arxiv.org/abs/2412.00300)** (*2024*)
  > Presents a feedback - driven plan critic, optimizing plans via RL with human feedback and GA, bridging gaps in planner research.

- **[Enhancing Robot Task Planning: Integrating Environmental Information and Feedback Insights through Large Language Models](https://ieeexplore.ieee.org/abstract/document/10661782)** (*2024*)
  > Presents EnviroFeedback Planner, integrating environmental info into prompt building and feedback for better agent execution in task planning.

- **[Devil's Advocate: Anticipatory Reflection for LLM Agents](https://arxiv.org/abs/2405.16334)** (*2024*)
  > A novel approach equips LLM agents with introspection, prompting task decomposition, continuous self - assessment, and three - fold intervention for better consistency and adaptability.

- **[Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios](https://arxiv.org/abs/2401.17167)** (*2024*)
  > Presents UltraTool, a benchmark for LLMs in real - world tool use. It evaluates the whole process, independently assesses planning, and removes pre - defined toolset restrictions.

- **[On the Structural Memory of LLM Agents](https://arxiv.org/abs/2412.15266)** (*2024*)
  > This paper explores how memory structures and retrieval methods affect LLM - based agents, finding task - specific advantages and iterative retrieval's superiority.

- **[CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](https://arxiv.org/abs/2303.17760)** (*2023*)
  > Proposes a role - playing communicative agent framework, offers scalable study approach for multi - agent systems, and open - sources library.

- **[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)** (*2023*)
  > AutoGen is an open - source framework for LLM apps. It enables customizable multi - agent conversation, flexible programming, and building diverse apps.

- **[AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation](https://arxiv.org/abs/2312.13010)** (*2023*)
  > The paper introduces AgentCoder, a multi - agent framework for code generation. It addresses balancing issues and outperforms existing methods.

- **[War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars](https://arxiv.org/abs/2311.17227)** (*2023*)
  > Proposes WarAgent, an LLM - powered multi - agent system for simulating historical conflicts, offering new insights for conflict resolution and peacekeeping.

- **[Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Abstract-Conference.html)** (*2023*)
  > The paper studies Minecraft planning for multi - task agents. It identifies two challenges and proposes a method to address inefficient planning.

- **[TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage](https://arxiv.org/abs/2308.03427)** (*2023*)
  > Presents a framework for LLM - based AI agents, designs two agent types, evaluating TPTU abilities to guide LLM use in AI apps.

### Agent Evolution

- **[Evolutionary optimization of model merging recipes](https://www.nature.com/articles/s42256-024-00975-8)** (*2025*)
  > Proposes an evolutionary approach for model merging, operating in two spaces, enabling cross - domain merging and introducing a new model composition paradigm.

- **[CREAM: Consistency Regularized Self-Rewarding Language Models](https://openreview.net/pdf?id=Vf6RDObyEF)** (*2025*)
  > This paper formulates a framework for self - rewarding LLM, introduces regularization, and proposes CREAM to use reward consistency for more reliable data.

- **[KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/pdf/2403.03101)** (*2025*)
  > The paper presents KNOWAGENT, using action knowledge base and self - learning to enhance LLM planning and mitigate hallucinations.

- **[STeCa: Step-level Trajectory Calibration for LLM Agent Learning](https://arxiv.org/abs/2502.14276)** (*2025*)
  > Paper proposes STeCa, a framework for LLM agent learning. It constructs calibrated trajectories via step - level reward comparison and LLM reflection.

- **[SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks](https://arxiv.org/abs/2503.15478)** (*2025*)
  > Introduced ColBench benchmark. Proposed SWEET - RL, using training - time info for critic model, offering step - level rewards to optimize LLM agents.

- **[DualRAG: A Dual-Process Approach to Integrate Reasoning and Retrieval for Multi-Hop Question Answering](https://arxiv.org/abs/2504.18243)** (*2025*)
  > The paper proposes DualRAG, a dual - process framework integrating reasoning and retrieval for MHQA. Its coupled processes form a cycle and work well across scales.

- **[Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward](https://arxiv.org/abs/2508.12800)** (*2025*)
  > Proposes Atomic Thought paradigm and Atom - Searcher RL framework, integrating thought units and rewards for better agentic deep research with unique supervision and reasoning.

- **[PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic Reasoning](https://arxiv.org/abs/2508.21104)** (*2025*)
  > Proposes PVPO, a RL method with advantage reference anchor and pre - sampling. Corrects bias, cuts rollout reliance, and selects high - gain data.

- **[Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation](https://arxiv.org/pdf/2402.11443)** (*2024*)
  > A benchmark self - evolving multi - agent framework extends benchmarks, uses six operations for fine - grained LLM evaluation, aiding model selection.

- **[Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization](https://aclanthology.org/2024.acl-long.292.pdf)** (*2024*)
  > Proposes Agent-Pro, an LLM-based agent using policy-level reflection and optimization. It evolves via dynamic belief process and DFS for better policies.

- **[Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf)** (*2024*)
  > The paper proposes CORY, extending LLM fine - tuning to a multi - agent framework. Agents coevolve, potentially superior to PPO for real - world refinement.

- **[A Survey on Self-Evolution of Large Language Models](https://arxiv.org/pdf/2404.14387)** (*2024*)
  > Presents a framework for LLM self - evolution with four phases. Categorizes objectives, summarizes literature, and points out challenges and future directions.

- **[LLM-Evolve: Evaluation for LLM’s Evolving Capability on Benchmarks](https://aclanthology.org/2024.emnlp-main.940.pdf)** (*2024*)
  > This paper proposes LLM-Evolve, an innovative framework extending benchmarks to sequential settings, enabling LLMs to learn from past experiences.

- **[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://openreview.net/pdf?id=Sx038qxjek)** (*2024*)
  > The paper introduces CRITIC, a framework enabling LLMs to self - correct via tool interaction, highlighting external feedback's role in LLMs' self - improvement.

- **[Iterative Translation Refinement with Large Language Models](https://aclanthology.org/2024.eamt-1.17.pdf)** (*2024*)
  > The paper proposes iterative prompting of LLMs for self - correcting translations. It emphasizes source - anchoring and shows improved human - perceived quality.

- **[Agent Alignment in Evolving Social Norms](https://arxiv.org/pdf/2401.04620)** (*2024*)
  > Proposes EvolutionaryAgent, an evolutionary framework for agent alignment. Transforms alignment into evolution/selection, applicable to various LLMs.

- **[Mitigating the Alignment Tax of RLHF](https://aclanthology.org/2024.emnlp-main.35.pdf)** (*2024*)
  > The paper reveals alignment tax in RLHF. It proposes HMA via model averaging to balance alignment and forgetting, maximizing performance with minimal tax.

- **[Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020)** (*2024*)
  > Paper studies Self - Rewarding LMs using LLM - as - a - Judge to self - reward during training, opening door for continuous improvement.

- **[V-STaR: Training Verifiers for Self-Taught Reasoners](https://openreview.net/pdf?id=stmqBSW2dV)** (*2024*)
  > Proposes V - STaR to train a verifier using both correct and incorrect self - generated solutions, improving solution selection and reasoning ability.

- **[RLCD: Reinforcement learning from contrastive distillation for LM alignment](https://openreview.net/pdf?id=v3XXtxWKi6)** (*2024*)
  > Proposes RLCD, a method for LM alignment without human feedback. Creates preference pairs via contrasting prompts to train a preference model.

- **[LANGUAGE MODEL SELF-IMPROVEMENT BY REIN- FORCEMENT LEARNING CONTEMPLATION](https://openreview.net/pdf?id=38E4yUbrgr)** (*2024*)
  > This paper presents RLC, a novel LMSI method leveraging evaluation - generation gap. It improves models without supervision and has broad applicability.

- **[ProAgent: Building Proactive Cooperative Agents with Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29710/31219)** (*2024*)
  > Proposes ProAgent, an LLM - based framework for proactive agents. It can adapt behavior, analyze states, infer intentions, and is modular, address zero - shot issues.

- **[Agent Planning with World Knowledge Model](https://openreview.net/pdf?id=j6kJSS9O6I)** (*2024*)
  > Presents parametric World Knowledge Model (WKM) for agent planning, synthesizing knowledge and guiding global & local planning, shows unique potential.

- **[Refining Guideline Knowledge for Agent Planning Using Textgrad](https://www.computer.org/csdl/proceedings-article/ickg/2024/088200a102/24sKrMSCxr2)** (*2024*)
  > This paper introduces Textgrad to optimize Guideline Knowledge for agents' embodied tasks, enabling auto - optimization via text gradients and failed trajectory analysis.

- **[Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://arxiv.org/abs/2305.19118)** (*2024*)
  > This paper proposes a Multi - Agent Debate framework to solve the Degeneration - of - Thought problem in LLMs, encouraging divergent thinking.

- **[LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error](https://aclanthology.org/2024.acl-long.570/)** (*2024*)
  > Existing LLMs' tool - use accuracy is low. A novel simulated trial - and - error method is proposed, inspired by biological systems, for better tool learning.

- **[AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf)** (*2023*)
  > AlpacaFarm addresses challenges in LLM development. It simulates feedback cheaply, offers evaluation and method implementations, validating end - to - end.

- **[SELF-REFINE:
 Iterative Refinement with Self-Feedback](https://openreview.net/pdf?id=S37hOerQLB)** (*2023*)
  > Introduces SELF - REFINE, an approach for iterative LLM output refinement without extra training, demonstrating test - time improvement of LLMs.

- **[Self-Evolution Learning for Discriminative Language Model Pretraining](https://aclanthology.org/2023.findings-acl.254.pdf)** (*2023*)
  > Presents Self - Evolution learning (SE), a method for token masking and learning. Exploits data knowledge and uses novel smoothing, improving linguistic learning.

- **[Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning](https://arxiv.org/pdf/2311.08182)** (*2023*)
  > The paper introduces DIVERSEEVOL, a self-evolving mechanism for label-efficient instruction tuning, enhancing data diversity without human/LLM intervention.

- **[SELFEVOLVE: A Code Evolution Framework via Large Language Models](https://arxiv.org/pdf/2306.02907)** (*2023*)
  > Proposes SELF-EVOLVE, a two - step pipeline using LLMs as knowledge providers and self - reflective programmers, with no need for special test cases.

- **[SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754.pdf)** (*2023*)
  > Introduces SELF - INSTRUCT, a framework to boost language models' instruction - following via self - generated samples, almost annotation - free.

- **[Large Language Models are Better Reasoners with Self-Verification](https://aclanthology.org/2023.findings-emnlp.167.pdf)** (*2023*)
  > The paper proposes LLMs have self - verification abilities. It uses backward verification, taking CoT conclusions as conditions, to improve reasoning performance.

- **[CODET: CODE GENERATION WITH GENERATED TESTS](https://openreview.net/pdf?id=ktrw68Cmu9c)** (*2023*)
  > The paper proposes CODET, a method using pre - trained LMs to auto - generate test cases for code samples, facilitating better solution selection.

- **[Evolving Diverse Red-team Language Models in Multi-round Multi-agent Games](https://arxiv.org/pdf/2310.00322)** (*2023*)
  > Introduces dynamic Red Team Game to analyze multi - round interactions, develops GRTS to mitigate mode collapse, paves way for LLM safety.

- **[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)** (*2023*)
  > A multi - agent debate approach for LLMs is proposed. It enhances reasoning, factuality, is applicable to black - box models, and has potential for LLM advancement.

- **[CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](https://arxiv.org/pdf/2303.17760)** (*2023*)
  > Paper proposes role - playing framework for autonomous agent cooperation, offers scalable study approach, and open - sources library.

- **[STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning](https://openreview.net/pdf?id=_3ELRdg2sgI)** (*2022*)
  > Proposes STaR, a technique leveraging few rationale examples and rationale - free data to bootstrap complex reasoning, letting models learn from self - generated reasoning.

### Applications

- **[An active inference strategy for prompting reliable responses from large language models in medical practice](https://doi.org/10.1038/s41746-025-01516-2)** (*2025*)
  > The paper proposes a domain-specific dataset and an active inference-based prompting protocol to address LLM issues, enabling its safe medical integration.

- **[An evaluation framework for clinical use of large language models in patient interaction tasks](https://doi.org/10.1038/s41591-024-03328-5)** (*2025*)
  > Presents CRAFT - MD, an approach using natural dialogues for LLM clinical evaluation. Proposes recommendations for future LLM eval to enhance medical practice.

- **[Large Language Models lack essential metacognition for reliable medical reasoning](https://doi.org/10.1038/s41467-024-55628-6)** (*2025*)
  > Developed MetaMedQA to evaluate models' metacognition in medical reasoning, revealing deficiencies, emphasizing need for metacognition - based frameworks.

- **[Balancing autonomy and expertise in autonomous synthesis laboratories](https://doi.org/10.1038/s43588-025-00769-x)** (*2025*)
  > Comment on barriers in autonomous synthesis labs, propose human on - the - loop approach, and strategies for optimizing labs' features.

- **[SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation](https://arxiv.org/pdf/2504.12722)** (*2025*)
  > Introduces SimUSER, an agent framework using personas for cost - effective user simulation in recommender system eval., refines params for real - world engagement.

- **[Swarm Autonomy: From Agent Functionalization to Machine Intelligence](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adma.202312956)** (*2025*)
  > This review summarizes synthetic swarms from agent basics to applications, discussing emergent machine intelligence for real - world autonomous swarm design.

- **[ShowUI: One Vision-Language-Action Model for GUI Visual Agent](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_ShowUI_One_Vision-Language-Action_Model_for_GUI_Visual_Agent_CVPR_2025_paper.html)** (*2025*)
  > A vision-language-action model for GUI visual agents with UI-guided token selection, interleaved streaming, and curated datasets advances GUI assistance.

- **[Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227)** (*2025*)
  > The paper introduces Agent Laboratory, an LLM - based framework for full - cycle research. It reduces costs, and user feedback improves quality, accelerating discovery.

- **[Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents](https://arxiv.org/abs/2503.24047)** (*2025*)
  > The paper reviews LLM - based scientific agents, highlights differences from general agents, and offers a roadmap for scientific discovery.

- **[CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation](https://arxiv.org/abs/2506.21805)** (*2025*)
  > The paper presents CitySim, an urban simulator using LLMs. It uses recursive value - driven approach and endows agents with key features, enabling scalable urban studies.

- **[A Survey of AI for Materials Science: Foundation Models, LLM Agents, Datasets, and Tools](https://arxiv.org/abs/2506.20743)** (*2025*)
  > This paper surveys FMs in MatSci, introducing a taxonomy, discussing advances, reviewing resources, assessing pros & cons, and suggesting future directions.

- **[An Auditable Agent Platform For Automated Molecular Optimisation](https://www.arxiv.org/abs/2508.03444)** (*2025*)
  > A hierarchical agent framework automates molecular optimisation, creates auditable reasoning trajectories, and converts LLMs into auditable design systems.

- **[PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](https://arxiv.org/abs/2508.21720)** (*2025*)
  > A training - free PosterForest framework is proposed. It uses Poster Tree and multi - agent collaboration for poster generation, addressing structure and integration challenges.

- **[Automated Clinical Problem Detection from SOAP Notes using a Collaborative Multi-Agent LLM Architecture](https://arxiv.org/abs/2508.21803)** (*2025*)
  > A collaborative multi - agent system (MAS) mimicking a clinical team analyzes SOAP notes' S&O sections, offering a path to better clinical decision support tools.

- **[Think in Games: Learning to Reason in Games via Reinforcement Learning with Large Language Models](https://arxiv.org/abs/2508.21365)** (*2025*)
  > Proposes Think in Games (TiG) framework enabling LLMs to gain procedural knowledge via game - env interaction, bridging knowledge gaps and enhancing transparency.

- **[AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)** (*2025*)
  > The paper presents AlphaEvolve, an evolutionary coding agent. It autonomously improves algorithms, discovers new ones, and broadens automated discovery scope.

- **[Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227)** (*2025*)
  > Introduces Agent Laboratory, an LLM - based framework for full - cycle research. It cuts costs, benefits from human feedback, and frees researchers for ideation.

- **[Motif: Intrinsic Motivation from Artificial Intelligence Feedback](https://arxiv.org/pdf/2310.00166)** (*2024*)
  > Paper proposes Motif, a method to interface LLM prior knowledge with agents via intrinsic rewards, yielding intuitive behaviors and progress on tough tasks.

- **[Baba Is AI: Break the Rules to Beat the Benchmark](https://arxiv.org/pdf/2407.13729)** (*2024*)
  > This paper likely presents a novel approach in action games under “Applications” section, with potential rule - breaking strategies for agents.

- **[Large language model-empowered agents for simulating macroeconomic activities](https://aclanthology.org/2024.acl-long.829/)** (*2024*)
  > This paper uses large language model-empowered agents to simulate macro - economic activities, offering a novel approach in economic applications.

- **[CompeteAI: Understanding the Competition Dynamics in Large Language Model-based Agents](https://arxiv.org/abs/2310.17512)** (*2024*)
  > The paper focuses on competition dynamics in LLMs-based agents in Economy applications, offering novel insights for the field.

- **[Understanding the benefits and challenges of using large language model-based conversational agents for mental well-being support](https://pmc.ncbi.nlm.nih.gov/articles/PMC10785945/)** (*2024*)
  > The paper explores benefits and challenges of large language model - based conversational agents for mental well - being support in psychology applications.

- **[Exploring Collaboration Mechanisms for LLM Agents](https://aclanthology.org/2024.acl-long.782/)** (*2024*)
  > This paper explores collaboration mechanisms for LLM agents in the psychology applications, bringing novel ideas to large model - based agents.

- **[Simulating Human Society with Large Language Model Agents: City, Social Media, and Economic System](https://dl.acm.org/doi/10.1145/3589335.3641253)** (*2024*)
  > The paper applies large language model agents to simulate human society, covering city, social media, and economic systems, a novel contribution in the field.

- **[Can large language models transform computational social science?](https://aclanthology.org/2024.cl-1.8/)** (*2024*)
  > Paper explores if large language models can transform computational social science in social applications, offering novel insights.

- **[AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems](https://arxiv.org/pdf/2310.09233)** (*2024*)
  > Proposes AgentCF to simulate user-item interactions. Considers users and items as agents, uses collaborative learning to model two-sided relations, inspiring behavior simulation.

- **[On Generative Agents in Recommendation](https://arxiv.org/abs/2310.10108)** (*2024*)
  > Proposes Agent4Rec, an LLM - empowered user simulator for recommenders with profile, memory, and action modules, exploring human behavior simulation.

- **[ChatDev: Communicative Agents for Software Development](https://aclanthology.org/2024.acl-long.810/)** (*2024*)
  > Introduces ChatDev, a chat - powered dev framework. LLMs agents use unified language comm. in design, coding, testing, bridging phases via lang. https://github.com/OpenBMB/ChatDev

- **[CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing Experiments](https://arxiv.org/abs/2404.18021)** (*2024*)
  > Introduces CRISPR - GPT, an LLM agent with domain knowledge and tools for auto - designing gene - editing experiments, addresses ethics, bridges researcher - technique gap.

- **[SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adma.202413523)** (*2024*)
  > SciAgents uses ontological KGs, LLMs, and multi - agent systems to uncover interdisciplinary relations in biomaterials, autonomously generating and refining hypotheses for discovery.

- **[Medical large language models are susceptible to targeted misinformation attacks](https://doi.org/10.1038/s41746-024-01282-7)** (*2024*)
  > The paper reveals LLMs in medicine are vulnerable. Just 1.1% weight manipulation can inject incorrect facts, stressing need for security measures.

- **[CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis](https://arxiv.org/abs/2407.09811)** (*2024*)
  > Introduces CellAgent, an LLM - driven multi - agent framework for scRNA - seq analysis. It has expert roles, decision - making and self - iterative mechanisms, reducing workload.

- **[Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents](https://arxiv.org/pdf/2302.01560)** (*2023*)
  > Paper proposes “DEPS”, an interactive planning approach with LLMs for multi - task agents, refining plans and showing effectiveness across domains.

- **[Language Models Meet World Models: Embodied Experiences Enhance Language Models](https://arxiv.org/abs/2305.10626.pdf)** (*2023*)
  > Combines language & world models, using embodied experiences. Applicable in simulation games, enhancing model capabilities.

- **[ChessGPT: Bridging Policy Learning and Language Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/16b14e3f288f076e0ca73bdad6405f77-Abstract-Datasets_and_Benchmarks.html)** (*2023*)
  > The paper bridges policy learning and language modeling, with potential applications in competition games, offering a novel approach in large model - based agents.

- **[Mindagent: Emergent gaming interaction](https://arxiv.org/pdf/2309.09971)** (*2023*)
  > This paper explores emergent gaming interaction in cooperation games, presenting novel applications for large model - based agents.

- **[Exploring large language models for communication games: An empirical study on Werewolf](https://arxiv.org/abs/2309.04658)** (*2023*)
  > This paper empirically explores large language models in Werewolf, a communication game, contributing novel applications in this area.

- **[Language as reality: a co-creative storytelling game experience in 1001 nights using generative AI](https://ojs.aaai.org/index.php/AIIDE/article/view/27539)** (*2023*)
  > This paper presents a co - creative storytelling game in "1001 Nights" via generative AI, contributing fresh application in game generation.

- **[TradingGPT: Multi-Agent System with Layered Memory and Distinct Characters for Enhanced Financial Trading Performance](https://arxiv.org/abs/2309.03736)** (*2023*)
  > TradingGPT presents a multi - agent system with layered memory and distinct characters to boost financial trading performance, a novel approach in the field.

- **[Using large language models to simulate multiple humans and replicate human subject studies](https://proceedings.mlr.press/v202/aher23a/aher23a.pdf)** (*2023*)
  > The paper applies large language models to simulate humans for replicating subject studies, contributing novel methods in psychological applications.

- **[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)** (*2023*)
  > Paper presents generative agents simulating human behavior, with novelty in application to society, a potential addition to large model - based agents study.

- **[Self-collaboration Code Generation via ChatGPT](https://arxiv.org/abs/2304.07590)** (*2023*)
  > A self - collaboration framework for code generation using LLMs like ChatGPT is proposed. It forms virtual teams of agents, improving complex task handling without human intervention.

- **[Language models can solve computer tasks](https://openreview.net/pdf?id=M6OmjAZ4CX)** (*2023*)
  > The paper presents RCI, a simple prompting scheme enabling pre - trained LLMs to execute computer tasks via natural language, enhancing reasoning and outperforming other methods.

- **[ChemCrow: Augmenting large-language models with chemistry tools](https://arxiv.org/abs/2304.05376)** (*2023*)
  > Introduces ChemCrow, an LLM chemistry agent integrating 18 tools. Augments LLM in chemistry, automates tasks, and bridges experimental and computational chemistry.

- **[AlphaFlow: autonomous discovery and optimization of multi-step chemistry using a self-driven fluidic lab guided by reinforcement learning](https://www.nature.com/articles/s41467-023-37139-y)** (*2023*)
  > The paper presents AlphaFlow, a self - driven fluidic lab using reinforcement learning for multi - step chemistry discovery, demonstrating its potential for new synthetic routes beyond cALD.

- **[Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://proceedings.mlr.press/v162/huang22a.html)** (*2022*)
  > This paper presents language models as zero - shot planners for embodied agents, with applications in simulation games, offering actionable knowledge extraction novelty.

- **[Stress-testing the resilience of the Austrian healthcare system using agent-based simulation](https://doi.org/10.1038/s41467-022-31766-7)** (*2022*)
  > A data - driven agent - based framework quantifies regional healthcare resilience to shocks, helps identify care access bottlenecks and relates systemic to individual indicators.

### Datasets & Benchmarks

- **[AgentHarm: Benchmarking Robustness of LLM Agents on Harmful Tasks](https://openreview.net/pdf?id=AC5n7xHuR1)** (*2025*)
  > Proposes AgentHarm, a new benchmark for LLM agents' robustness. Covers 11 harm categories, enabling evaluation of attacks and defenses.

- **[AI Hospital: Benchmarking Large Language Models in a Multi-agent Medical Interaction Simulator](https://aclanthology.org/2025.coling-main.680.pdf)** (*2025*)
  > Introduced AI Hospital for simulating medical interactions, developed MVME benchmark, proposed dispute - resolution mechanism to boost LLMs' clinical abilities.

- **[Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation](https://aclanthology.org/2025.coling-main.223.pdf)** (*2025*)
  > A multi - agent benchmark self - evolving framework dynamically evaluates LLMs. It reframes instances and extends datasets, aiding model selection and benchmark evolution.

- **[DCA-Bench: A Benchmark for Dataset Curation Agents](https://openreview.net/pdf?id=a4sknPttwV)** (*2025*)
  > The paper sets up a benchmark for LLM agents to detect wild dataset issues, curates test cases, and proposes an evaluation framework, promoting real - world curation.

- **[MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents](https://arxiv.org/pdf/2501.14654)** (*2025*)
  > The paper introduces MedAgentBench, a benchmark for medical LLM agents with clinically - derived tasks and realistic data, enabling evaluation and optimization in medical domain.

- **[MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://openreview.net/pdf?id=6s5uXNWGIh)** (*2025*)
  > Introduced MLE - bench for evaluating AI agents in ML engineering. Curated tasks, set baselines, evaluated models, and opened - sourced code for future research.

- **[EgoLife: Towards Egocentric Life Assistant](https://arxiv.org/pdf/2503.03803)** (*2025*)
  > Introduced EgoLife project for egocentric life assistant. Created EgoLife Dataset and EgoLifeQA tasks for daily life assistance.

- **[DSBench: How Far Are Data Science Agents to Becoming Data Science Experts?](https://arxiv.org/abs/2409.07703)** (*2025*)
  > The paper introduces DSBench, a comprehensive benchmark for data science agents with realistic tasks, bridging the gap between benchmarks and real - world apps.

- **[Towards Internet-Scale Training For Agents](https://arxiv.org/abs/2502.06776)** (*2025*)
  > The paper develops a pipeline for Internet-scale agent training without extensive human annotations, with LLMs handling task gen., execution, and review.

- **[macOSWorld: An Interactive Benchmark for GUI Agents](https://arxiv.org/abs/2506.04135)** (*2025*)
  > Paper presents macOSWorld, first macOS GUI agent benchmark with multilingual tasks and safety subset, bridging OS evaluation gaps.

- **[Humanity's Last Exam](https://arxiv.org/abs/2501.14249)** (*2025*)
  > Introduces Humanity's Last Exam (HLE), a multi - modal, broad - coverage benchmark for LLMs, gap shown, and publicly released for research.

- **[MCPEval: Automatic MCP-based Deep Evaluation for AI Agent Models](https://arxiv.org/abs/2507.12806)** (*2025*)
  > Introduces MCPEval, an MCP - based open - source framework for automating LLM agent evaluation, standardizing metrics and reducing manual work.

- **[IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis](https://arxiv.org/abs/2505.18223)** (*2025*)
  > Introduces IDA - Bench, a novel benchmark for LLMs in multi - round data analysis, emphasizing balance between instruction - following and reasoning.

- **[SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks](https://arxiv.org/abs/2506.11791)** (*2025*)
  > Introduces SEC - bench, an automated benchmarking framework for LLM agents on real - world security tasks, with a novel multi - agent scaffold to create datasets.

- **[MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents](https://arxiv.org/abs/2508.21475)** (*2025*)
  > Introduces MMSearch - Plus benchmark for multimodal browsing agents, with novel curation and agent framework, addressing genuine multimodal challenges.

- **[MultiAgentBench : Evaluating the Collaboration and Competition of LLM agents](https://aclanthology.org/2025.acl-long.421/)** (*2025*)
  > This paper introduces MultiAgentBench for evaluating multi - agent systems in diverse scenarios, assesses protocols and strategies like cognitive planning, and will release code and data.

- **[Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)** (*2025*)
  > Many agentic benchmarks have setup or reward issues. The paper introduces ABC guidelines to make agentic evaluation rigorous.

- **[AgentBench: Evaluating LLMs as Agents](https://openreview.net/pdf?id=zAdUB0aCTQ)** (*2024*)
  > Presents AgentBench with 8 environments to evaluate LLM agents, identifies failure reasons, and offers improvement strategies like multi - round alignment training.

- **[AgentQuest: A Modular Benchmark Framework to Measure Progress and Improve LLM Agents](https://aclanthology.org/2024.naacl-demo.19.pdf)** (*2024*)
  > Proposes AgentQuest, a framework with modular benchmarks/metrics and two new evaluation metrics for tracking LLM agent progress.

- **[BENCHAGENTS: Automated Benchmark Creation with Agent Interaction](https://arxiv.org/pdf/2410.22584)** (*2024*)
  > Introduces BENCHAGENTS, an LLM - based framework to automate benchmark creation for complex capabilities, ensuring data quality with agent interaction and human feedback.

- **[Benchmarking Data Science Agents](https://aclanthology.org/2024.acl-long.308.pdf)** (*2024*)
  > Presents DSEval, a novel evaluation paradigm and benchmarks for data science agents, with a bootstrapped method for better coverage and comprehensiveness.

- **[Benchmarking Large Language Models as AI Research Agents](https://openreview.net/pdf?id=N9wD4RFWY0)** (*2024*)
  > Proposes MLAgent-Bench for benchmarking AI research agents, designs an LLM-based agent, and identifies key challenges for such agents.

- **[Benchmarking Large Language Models for Multi-agent Systems: A Comparative Analysis of AutoGen, CrewAI, and TaskWeaver](https://link.springer.com/chapter/10.1007/978-3-031-70415-4_4)** (*2024*)
  > The paper benchmarks three LLMs-powered multi - agent systems (AutoGen, CrewAI, TaskWeaver) on ML code gen, advancing collaborative problem - solving research.

- **[BLADE- Benchmarking Language Model Agents](https://aclanthology.org/2024.findings-emnlp.815.pdf)** (*2024*)
  > This paper presents BLADE, a benchmark for automatically evaluating agents' multifaceted approaches to open - ended research, enabling agent evaluation for data - driven science.

- **[CRAB: Cross-platfrom agent benchmark for multi-modal embodied language model agents](https://openreview.net/pdf?id=kyExS4V0H7)** (*2024*)
  > Introduced Crab, a cross - environment agent benchmark framework with graph - based eval. method and task construction mechanism, and developed Crab Benchmark - v0.

- **[CToolEval: A Chinese Benchmark for LLM-Powered Agent Evaluation in Real-World API Interactions](https://aclanthology.org/2024.findings-acl.928.pdf)** (*2024*)
  > Propose CToolEval benchmark for Chinese LLM agents with 398 APIs. Present an evaluation framework and release data/codes to promote agent - level research.

- **[DA-Code: Agent Data Science Code Generation Benchmark for Large Language Models](https://aclanthology.org/2024.emnlp-main.748.pdf)** (*2024*)
  > Introduces DA - Code, a code gen benchmark for LLMs on agent - based data science. It has unique tasks, real data, and requires complex langs, released on GitHub.

- **[Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b631da756d1573c24c9ba9c702fde5a9-Abstract-Datasets_and_Benchmarks_Track.html)** (*2024*)
  > Proposes an Embodied Agent Interface to unify tasks, modules, and metrics, comprehensively assessing LLMs for embodied decision making.

- **[GTA: A Benchmark for General Tool Agents](https://proceedings.neurips.cc/paper_files/paper/2024/file/8a75ee6d4b2eb0b777f549a32a5a5c28-Paper-Datasets_and_Benchmarks_Track.pdf)** (*2024*)
  > Proposes GTA, a benchmark for general tool agents with real queries, deployed tools, and multimodal inputs to assess LLMs' real - world problem - solving.

- **[LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs](http://javascript:void())** (*2024*)
  > Presents LaMPilot, integrating LLMs into AD for following user instructions. Introduces LaMPilot - Bench and releases code/data for further research.

- **[ML Research Benchmark](https://arxiv.org/pdf/2410.22553)** (*2024*)
  > The paper presents ML Research Benchmark with 7 tasks for AI agents, offering a framework to assess them on real - world research challenges.

- **[MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains](https://arxiv.org/pdf/2407.18961)** (*2024*)
  > The paper introduces MMAU benchmark, with offline tasks across 5 domains and 5 capabilities, enhancing interpretability of LLM agents.

- **[OmniACT: A Dataset and Benchmark for Enabling Multimodal Generalist Autonomous Agents for Desktop and Web](https://arxiv.org/pdf/2402.17553)** (*2024*)
  > Introduces OmniACT, a first - of - its - kind dataset and benchmark for assessing agents' ability in computer task automation, covering desktop apps.

- **[OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d413e48f84dc61244b6be550f1cd8f5-Paper-Datasets_and_Benchmarks_Track.pdf)** (*2024*)
  > Introduces OSWorld, a scalable real computer env. for multimodal agents, creates 369 - task benchmark, aids multimodal generalist agent development.

- **[Revisiting Benchmark and Assessment: An Agent-based Exploratory Dynamic Evaluation Framework for LLMs](https://arxiv.org/pdf/2410.11507)** (*2024*)
  > The paper introduces Benchmark+ and Assessment+, proposes TestAgent framework, enabling dynamic benchmark generation and domain - adaptive assessments for LLMs.

- **[Seal-Tools: Self-instruct Tool Learning Dataset for Agent Tuning and Detailed Benchmark](https://arxiv.org/pdf/2405.08355)** (*2024*)
  > A new tool learning dataset Seal - Tools with self - instruct method for generation, hard instances, strict metrics as a new benchmark for LLMs' tool - calling.

- **[Tapilot-Crossing: Benchmarking and Evolving LLMs Towards Interactive Data Analysis Agents](https://arxiv.org/pdf/2403.05307)** (*2024*)
  > The paper introduces Tapilot - Crossing benchmark for evaluating LLM agents in interactive data analysis and proposes AIR strategy to evolve LLMs into effective agents.

- **[TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/pdf/2412.14161)** (*2024*)
  > This paper introduces TheAgentCompany, an extensible benchmark for evaluating AI agents on real - world tasks, mimicking a software company environment.

- **[Tur[k]ingBench: A Challenge Benchmark for Web Agents](https://arxiv.org/pdf/2403.11905)** (*2024*)
  > Presents TurkingBench, a benchmark using natural HTML pages from crowdsourcing. Develops an evaluation framework to spur web - based agent progress.

- **[Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models](https://aclanthology.org/2024.findings-acl.557/)** (*2024*)
  > Paper identifies issues in agent training, proposes Agent - FLAN to fine - tune LLMs, decomposes corpus, uses negatives to reduce hallucinations.

- **[AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories](https://aclanthology.org/2024.findings-emnlp.116/)** (*2024*)
  > Introduces AgentBank, a large trajectory data collection. Uses novel annotation, fine - tunes LLMs to get Samoyed, shows data scaling for agent capabilities.

- **[AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning](http://arxiv.org/abs/2402.15506)** (*2024*)
  > Introduces AgentOhana to unify agent trajectories from diverse sources, enabling a balanced training pipeline, and presents xLAM-v0.1 for AI agents.

- **[AgentTuning: Enabling Generalized Agent Abilities for LLMs](https://aclanthology.org/2024.findings-acl.181/)** (*2024*)
  > This paper presents AgentTuning, a method using AgentInstruct and hybrid tuning to boost LLMs' agent abilities without sacrificing generality, and open-sources models.

- **[Executable Code Actions Elicit Better LLM Agents](https://proceedings.mlr.press/v235/wang24h.html)** (*2024*)
  > This paper proposes CodeAct, using Python code for LLM agents' actions. It creates a dataset and a finetuned agent with self - debugging for complex tasks.

- **[AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901)** (*2024*)
  > Built AppWorld Engine and Benchmark to address gaps in existing tool - use benchmarks, enabling rich and interactive code gen. for agents.

- **[SheetAgent: Towards A Generalist Agent for Spreadsheet Reasoning and Manipulation via Large Language Models](https://arxiv.org/abs/2403.03636)** (*2024*)
  > The paper introduces SheetRM benchmark and proposes SheetAgent, an LLM - based agent with three modules, enabling autonomous spreadsheet reasoning and manipulation.

- **[FireAct: Toward Language Agent Fine-tuning](http://arxiv.org/abs/2310.05915)** (*2023*)
  > The paper explores LM fine - tuning for language agents. It proposes FireAct using diverse data, revealing benefits and offering experimental insights.

### Ethics

- **[Medical large language models are vulnerable to data-poisoning attacks](https://www.nature.com/articles/s41591-024-03445-1)** (*2025*)
  > Paper assesses LLM data - poisoning attacks, finds low - ratio misinfo harms models, and proposes a graph - based mitigation strategy.

- **[Foundation Models and Fair Use](https://www.jmlr.org/papers/v24/23-0569.html)** (*2024*)
  > Discusses legal & ethical risks of foundation models on copyrighted data, suggests technical mitigations, and advocates law-tech co - evolution for fair use.

- **[Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model](https://www.jmlr.org/papers/v24/23-0069.html)** (*2023*)
  > The paper quantifies BLOOM's carbon footprint across its life - cycle and studies its inference emissions, discussing estimation challenges and future research.

- **[LLaMA: Open and Efficient Foundation Language Models](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)** (*2023*)
  > Introduces LLaMA models (7B - 65B params). Trains with public datasets only, and releases them to the research community.

- **[Predictability and Surprise in Large Generative Models](https://dl.acm.org/doi/abs/10.1145/3531146.3533229)** (*2022*)
  > This paper reveals large generative models' paradox of predictability and unpredictability, shows harms, and lists AI community interventions.

- **[On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://dl.acm.org/doi/10.1145/3442188.3445922)** (*2021*)
  > The paper questions the size of language models, explores associated risks, and offers mitigation recommendations beyond just scaling up.

- **[Process for Adapting Language Models to Society (PALMS) with Values-Targeted Datasets](https://proceedings.neurips.cc/paper_files/paper/2021/hash/2e855f9489df0712b4bd8ea9e2848c5a-Abstract.html)** (*2021*)
  > Proposes PALMS, an iterative process using values - targeted datasets to change language model behavior with a small, curated dataset.

- **[GPT-3: Its Nature, Scope, Limits, and Consequences](https://link.springer.com/article/10.1007/s11023-020-09548-1)** (*2020*)
  > Paper analyzes GPT - 3 via reversible/irreversible questions, presents three tests it fails, and outlines consequences of artefact industrialization.

- **[Energy and Policy Considerations for Modern Deep Learning Research](https://ojs.aaai.org/index.php/AAAI/article/view/7123)** (*2020*)
  > It reveals high costs of large neural network computation, quantifies NLP model costs, and offers recommendations for cost - reduction and equity.

- **[Defending Against Neural Fake News](https://proceedings.neurips.cc/paper/2019/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html)** (*2019*)
  > Presents Grover for controllable text gen to study neural fake - news risks, shows Grover's self - defense value, and discusses ethics.

### Security

- **[RTBAS: Defending LLM Agents Against Prompt Injection and Privacy Leakage](https://arxiv.org/pdf/2502.08966)** (*2025*)
  > Paper introduces RTBAS for TBAS, adapting Information Flow Control and using novel screeners to auto - handle tool calls, reducing user burden.

- **[Red-Teaming LLM Multi-Agent Systems via Communication Attacks](https://arxiv.org/pdf/2502.14847)** (*2025*)
  > Presents Agent-in-the-Middle (AiTM), a novel attack on LLM-MAS via message manipulation, showing need for multi - agent system security.

- **[Unveiling Privacy Risks in LLM Agent Memory](https://arxiv.org/abs/2502.13172)** (*2025*)
  > The paper proposes MEXTRA under black-box setting to extract private info from LLM agent memory, and explores factors of leakage, urging for safeguards.

- **[AEIA-MN: Evaluating the Robustness of Multimodal LLM-Powered Mobile Agents Against Active Environmental Injection Attacks](https://arxiv.org/pdf/2502.13053)** (*2025*)
  > The paper defines Active Environment Injection Attack (AEIA) and proposes AEIA - MN to evaluate MLLM - based agents' robustness against such threats.

- **[Firewalls to Secure Dynamic LLM Agentic Networks](https://arxiv.org/pdf/2502.01822)** (*2025*)
  > The paper identifies comm. props. for LLM agentic networks, proposes a design for balance, and constructs firewall rules via simulations.

- **[AUTOHIJACKER: AUTOMATIC INDIRECT PROMPT INJECTION AGAINST BLACK-BOX LLM AGENTS](https://openreview.net/pdf?id=2VmB01D9Ef)** (*2025*)
  > Proposes AutoHijacker, an automatic indirect black - box prompt injection attack. It uses LLM - as - optimizers, batch - based optimization, and trainable memory.

- **[AI Agents Under Threat: A Survey of Key Security Challenges and Future Pathways](https://dl.acm.org/doi/pdf/10.1145/3716628)** (*2025*)
  > This paper categorizes emerging security threats to AI agents into four knowledge gaps, aiming to inspire research for more secure agent apps.

- **[SAGA: A Security Architecture for Governing AI Agentic Systems](https://arxiv.org/abs/2504.21034)** (*2025*)
  > Proposes SAGA, a security architecture for governing agentic systems, offering user oversight and fine - grained access control, enabling secure agent deployment.

- **[WebInject: Prompt Injection Attack to Web Agents](https://arxiv.org/abs/2505.11717)** (*2025*)
  > Proposes WebInject, a prompt injection attack on web agents. Adds pixel perturbation, trains NN for mapping, and solves optimization problem.

- **[DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent](https://arxiv.org/abs/2502.12575)** (*2025*)
  > Proposes Dynamically Encrypted Multi - Backdoor Implantation Attack with dynamic encryption and sub - fragments to bypass safety audits. Also presents AgentBackdoorEval dataset.

- **[CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models](https://arxiv.org/abs/2502.14529)** (*2025*)
  > Presents Contagious Recursive Blocking Attacks (Corba) on LLM - MASs. Novel in propagation and resource - depletion, hard to mitigate by alignment.

- **[G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems](https://arxiv.org/abs/2502.11127)** (*2025*)
  > The paper introduces G - Safeguard for LLM - MAS. It uses graph neural networks for anomaly detection and topological intervention, adaptable and combinable with mainstream MAS.

- **[AgentHarm: Benchmarking Robustness of LLM Agents on Harmful Tasks](https://openreview.net/forum?id=AC5n7xHuR1)** (*2025*)
  > Proposes AgentHarm, a new benchmark for LLM agent misuse with 110 malicious tasks, enabling evaluation of attacks and defenses.

- **[Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks](https://arxiv.org/abs/2502.08586)** (*2025*)
  > This paper analyzes unique security and privacy vulnerabilities of LLM agents, provides an attack taxonomy, and conducts simple attacks, needing no ML knowledge.

- **[A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment](https://arxiv.org/abs/2504.15585)** (*2025*)
  > This paper introduces "full - stack" safety for LLMs, covering the whole lifecycle, with extensive literature and unique insights on research directions.

- **Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems** (*2025*)
  > The paper reveals LLM-to-LLM prompt injection in multi - agent systems, proposes Prompt Infection, and suggests LLM Tagging to mitigate it.

- **[LLM-based Multi-Agent Systems: Techniques and Business Perspectives](https://arxiv.org/pdf/2411.14033?)** (*2024*)
  > This paper explores LLM-based Multi-Agent Systems (LaMAS), presents its advantages, provides a protocol, and sees it as a solution for artificial collective intelligence.

- **[BlockAgents: Towards Byzantine-Robust LLM-Based Multi-Agent Coordination via Blockchain](https://dl.acm.org/doi/pdf/10.1145/3674399.3674445)** (*2024*)
  > The paper proposes BlockAgents, integrating blockchain into LLM-based multi-agent systems. It features PoT and multi-metric evaluation to mitigate Byzantine behaviors.

- **[PROMPT INFECTION: LLM-TO-LLM PROMPT INJECTION WITHIN MULTI-AGENT SYSTEMS](https://arxiv.org/pdf/2410.07283)** (*2024*)
  > Reveals LLM-to-LLM prompt injection in multi - agent systems, proposes “Prompt Infection”, and “LLM Tagging” defense to enhance security.

- **[AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://openreview.net/pdf?id=m1YYAQjO3w)** (*2024*)
  > Introduces AgentDojo, an extensible evaluation framework for AI agents on untrusted data, aiming to foster reliable and robust agent design.

- **[AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf)** (*2024*)
  > Proposes AGENTPOISON, a novel backdoor attack on LLM agents by poisoning memory/KB. No extra training, with good transferability and stealth.

- **[AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks](https://arxiv.org/pdf/2403.04783)** (*2024*)
  > Proposes AutoDefense, a multi - agent framework filtering LLM harmful responses, robust against attacks, enabling small LMs to defend larger ones.

- **[Imprompter- Tricking LLM Agents into Improper Tool Use](https://arxiv.org/pdf/2410.14923)** (*2024*)
  > The paper contributes to agent - based system security, presents obfuscated adversarial prompt attacks, and shows they work on multiple agents.

- **[TARGETING THE CORE: A SIMPLE AND EFFECTIVE METHOD TO ATTACK RAG-BASED AGENTS VIA DIRECT LLM MANIPULATION](https://arxiv.org/pdf/2412.04415)** (*2024*)
  > This paper reveals a critical LLM vulnerability via adversarial prefixes, highlighting need for multi - layered security in agent - based architectures.

- **[Prompt Injection as a Defense Against LLM-driven Cyberattacks](https://arxiv.org/pdf/2410.20911)** (*2024*)
  > Proposes Mantis, a defensive framework using prompt - injection to counter LLM - driven cyberattacks, can hack back attackers autonomously and is open - source.

- **[Evil Geniuses: Delving into the Safety of LLM-based Agents](https://arxiv.org/pdf/2311.11855)** (*2024*)
  > The paper explores LLM - based agent safety from three aspects. It proposes a template - based attack and "Evil Geniuses" method for in - depth analysis.

- **[AGENT SECURITY BENCH (ASB): FORMALIZING AND BENCHMARKING ATTACKS AND DEFENSES IN LLM-BASED AGENTS](https://arxiv.org/pdf/2410.02644?)** (*2024*)
  > Introduces Agent Security Bench (ASB) for formalizing, benchmarking LLM - based agent attacks/defenses, revealing vulnerabilities and future work.

- **[AGENTHARM: A BENCHMARK FOR MEASURING HARMFULNESS OF LLM AGENTS](https://arxiv.org/pdf/2410.09024)** (*2024*)
  > A new benchmark AgentHarm is proposed for LLM agent misuse research, with diverse malicious tasks and unique scoring criteria.

- **[CLAS 2024: The Competition for LLM and Agent Safety](https://openreview.net/pdf?id=GIDw94AlZK)** (*2024*)
  > CLAS 2024 advances LLM and agent safety understanding via three tracks, fostering community collaboration for safer AI systems.

- **[The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents](https://arxiv.org/pdf/2412.16682)** (*2024*)
  > The paper proposes reframing agent security to ensure task alignment. It develops Task Shield to verify instruction contribution to user goals.

- **[WIPI: A New Web Threat for LLM-Driven Web Agents](https://arxiv.org/pdf/2402.16965)** (*2024*)
  > This paper introduces a novel threat, WIPI, which indirectly controls Web Agents via web - page instructions, enhancing attack efficiency and stealth.

- **[Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast](https://arxiv.org/abs/2402.08567)** (*2024*)
  > Paper reveals 'infectious jailbreak' in multi - agent MLLM, shows its feasibility, and proposes a principle for defense spread restraint.

- **[CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models](https://arxiv.org/pdf/2502.14529)** (*2024*)
  > Introduces CORBA, a novel attack on LLM - MASs. It's contagious and recursive, hard to mitigate by alignment, effective across topologies and models.

- **[PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety](https://aclanthology.org/2024.acl-long.812/)** (*2024*)
  > This paper explores multi - agent system safety through agent psychology, proposes PsySafe framework, and offers insights for future research.

- **[Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In](https://arxiv.org/pdf/2410.16950)** (*2024*)
  > The paper introduces the foot - in - the - door attack on ReAct agents. It proposes a reflection mechanism to mitigate this security vulnerability.

- **[AGENT-SAFETYBENCH: Evaluating the Safety of LLM Agents](https://arxiv.org/pdf/2412.14470)** (*2024*)
  > The paper introduces AGENT - SAFETYBENCH to evaluate LLM agent safety, identifies flaws, and stresses need for advanced strategies, will release the benchmark.

- **[INJECAGENT: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents](https://arxiv.org/pdf/2403.02691)** (*2024*)
  > Introduces INJECAGENT, a benchmark for assessing IPI attack vulnerability of tool - integrated LLM agents, categorizing attack intents.

- **[PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety](https://arxiv.org/abs/2401.11880)** (*2024*)
  > This paper proposes PsySafe, a framework based on agent psychology, to address multi - agent system safety, offering insights into risk identification, evaluation, and mitigation.

- **[TrustAgent: Towards Safe and Trustworthy LLM-based Agents](https://arxiv.org/abs/2402.01586)** (*2024*)
  > Presents TrustAgent, an agent - constitution - based framework. Uses three strategies for LLM - agent safety, impacts helpfulness and reveals LLM reasoning importance.

- **[Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b6e9d6f4f3428cd5f3f9e9bbae2cab10-Abstract-Conference.html)** (*2024*)
  > This work formulates a framework for agent backdoor attacks, analyzes diverse forms, and reveals a need for targeted defenses against them.

- **[R-Judge: Benchmarking Safety Risk Awareness for LLM Agents](https://arxiv.org/abs/2401.10019)** (*2024*)
  > Introduces R-Judge, a benchmark for evaluating LLM agents' safety risk awareness, shows room for improvement, and reveals effective fine - tuning approach.

- **[NetSafe: Exploring the Topological Safety of Multi-agent Networks](https://arxiv.org/abs/2410.15686)** (*2024*)
  > This paper offers a topological view on multi - agent network safety, proposes NetSafe, identifies new phenomena, guiding future safety research.

- **[A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents](https://arxiv.org/abs/2402.10196)** (*2024*)
  > Presents first systematic mapping of adversarial attacks on language agents, with a framework and 12 scenarios, stressing risk - understanding urgency.

### Survey

- **[A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment](https://arxiv.org/abs/2504.15585)** (*2025*)
  > This paper first introduces "full - stack" safety for LLMs, covering the whole lifecycle, with rich literature and unique insights for future research.

- **[Trust but Verify! A Survey on Verification Design for Test-time Scaling](https://arxiv.org/abs/2508.16665)** (*2025*)
  > This survey covers diverse TTS verification approaches, presents a unified view of verifier training, filling a gap in the literature.

- **[A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers](https://arxiv.org/abs/2508.21148)** (*2025*)
  > This survey reframes Sci - LLMs development around data, analyzes datasets, evaluation, and proposes a shift to closed - loop systems for scientific discovery.

- **[Evaluation and Benchmarking of LLM Agents: A Survey](https://arxiv.org/abs/2507.21504)** (*2025*)
  > Presents a 2D taxonomy for LLM agent evaluation, highlights enterprise challenges, and identifies future research directions for systematic assessment.

- **[Benchmark Evaluations, Applications, and Challenges of Large Vision Language Models: A Survey](https://arxiv.org/pdf/2501.02189)** (*2025*)
  > This paper offers a systematic VLM overview: model info, architectures, benchmarks, applications, and challenges, with details in a GitHub repo.

- **[The Future is Agentic: Definitions, Perspectives, and Open Challenges of Multi-Agent Recommender Systems](https://arxiv.org/abs/2507.02097)** (*2025*)
  > This paper explores LLM agents in recommender systems, introducing a formalism, use - cases, challenges, and paves the way for next - gen services.

- **[Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks](https://arxiv.org/pdf/2502.08586)** (*2025*)
  > Analyzes unique security & privacy vulnerabilities of LLM agents, provides attack taxonomy, and conducts simple attacks on popular agents.

- **[Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/pdf/2501.06322)** (*2025*)
  > This paper surveys LLM-based Multi-Agent Systems, introduces a framework, explores applications, and identifies challenges and directions for AI collective intelligence.

- **[AI Agents Under Threat: A Survey of Key Security Challenges and Future Pathways](https://dl.acm.org/doi/abs/10.1145/3716628)** (*2025*)
  > This survey identifies four security knowledge - gaps for AI agents. It reviews threats, shows progress/limitations, and inspires future research.

- **[Large Model Based Agents: State-of-the-Art, Cooperation Paradigms, Security and Privacy, and Future Trends](https://arxiv.org/abs/2409.14457)** (*2024*)
  > Paper explores future autonomous collaboration of LM agents, covers current state, collaboration paradigms, security risks, and proposes future research directions.

- **[Agent AI: Surveying the Horizons of Multimodal Interaction](https://arxiv.org/abs/2401.03568)** (*2024*)
  > Defines "Agent AI" for interactive multimodal systems, explores action prediction, mitigates model hallucinations, and envisions virtual interactions.

- **[Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/abs/2402.01680)** (*2024*)
  > This survey discusses essential aspects and challenges of LLM-based multi-agent systems, provides datasets, and maintains a GitHub repo for latest research.

- **[Large Multimodal Agents: A Survey](https://arxiv.org/abs/2402.15116)** (*2024*)
  > Reviews LLM-driven multimodal agents, categorizes research, compiles evaluation methods, and proposes future directions.

- **[Understanding the planning of LLM agents: A survey](https://arxiv.org/abs/2402.02716)** (*2024*)
  > This survey offers the first systematic view of LLM - based agent planning, categorizes related works, analyzes directions, and discusses challenges.

- **[Computational Experiments Meet Large Language Model Based Agents: A Survey and Perspective](https://arxiv.org/abs/2402.00262)** (*2024*)
  > The paper explores combining computational experiments with LLM - based Agents, outlines their history, mutual advantages, and addresses challenges and trends.

- **[Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security](https://arxiv.org/abs/2401.05459)** (*2024*)
  > The paper focuses on Personal LLM Agents, discusses architecture, challenges, and solutions, envisioning them as a major software paradigm.

- **[Large Model Based Agents: State-of-the-Art, Cooperation Paradigms, Security and Privacy, and Future Trends](https://arxiv.org/abs/2409.14457)** (*2024*)
  > This paper explores future LM agent autonomous collaboration, covering current state, key tech, security & privacy, and suggests future research directions.

- **[The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey](https://arxiv.org/abs/2404.11584)** (*2024*)
  > This survey assesses AI agent implementations, sharing capabilities, insights, and design considerations, highlighting key themes for robust systems.

- **[Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects](https://arxiv.org/abs/2401.03428)** (*2024*)
  > This paper surveys LLM - based intelligent agents in single - and multi - agent systems, covering definitions, components, deployment mechanisms, and envisions their prospects.

- **[Position Paper: Agent AI Towards a Holistic Intelligence](https://arxiv.org/abs/2403.00833)** (*2024*)
  > Paper proposes Agent Foundation Model for embodied intelligence, discusses Agent AI's domain capabilities and interdisciplinary potential, guiding future research.

- **[Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://www.ijcai.org/proceedings/2024/0890.pdf)** (*2024*)
  > This survey delves into LLM - based multi - agent systems. It covers operation domains, agent profiles, and skill - development means. It also lists datasets and maintains a GitHub repo.

- **[LLM With Tools: A Survey](http://arxiv.org/abs/2409.18807)** (*2024*)
  > Presents a standardized tool - integration paradigm, explores challenges, innovative solutions, and the idea of LLMs creating tools, reproduces results.

- **[A Survey on the Memory Mechanism of Large Language Model based Agents](https://arxiv.org/abs/2404.13501)** (*2024*)
  > This paper comprehensively surveys LLM-based agents' memory mechanisms, reviewing design and evaluation, presenting applications, and suggesting future directions.

- **[Understanding the planning of LLM agents: A survey](https://arxiv.org/abs/2402.02716)** (*2024*)
  > This survey offers a systematic view of LLM - based agent planning, taxonomizes existing works, analyzes directions, and discusses research challenges.

- **[Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/pdf/2402.01680)** (*2024*)
  > This survey offers in - depth discussion on LLM - based multi - agent systems, their aspects, challenges, and provides datasets and an open - source repo.

- **[A Survey on Large Language Model-Based Game Agents](https://arxiv.org/pdf/2404.02039)** (*2024*)
  > Paper offers holistic overview of LLM - based game agents, details architecture, surveys agents across game genres, and presents future R & D directions.

- **[Large Language Models and Games: A Survey and Roadmap](https://arxiv.org/pdf/2402.18659)** (*2024*)
  > This paper surveys LLM applications in games, identifies LLM roles, discusses unexplored areas, and reconciles potential and limitations, paving the way for new research.

- **[Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects](https://arxiv.org/abs/2401.03428)** (*2024*)
  > This paper surveys LLM - based intelligent agents in single - and multi - agent systems, covering definitions, components, deployment, datasets, and envisions prospects.

- **[Navigating the Risks: A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents](https://arxiv.org/pdf/2411.09523?)** (*2024*)
  > This survey analyzes security, privacy, and ethics threats in LLM - based agents, proposes a taxonomy, and suggests future research directions.

- **[Security of AI Agents](https://arxiv.org/pdf/2406.08689)** (*2024*)
  > The paper identifies AI agents' security vulnerabilities from a system perspective, introduces defenses, and offers ways to enhance their safety and reliability.

- **[PERSONAL LLM AGENTS: INSIGHTS AND SURVEY ABOUT THE CAPABILITY, EFFICIENCY AND SECURITY](https://arxiv.org/pdf/2401.05459)** (*2024*)
  > The paper focuses on Personal LLM Agents, discusses their architecture, challenges, and presents solutions, envisioning them as a major software paradigm.

- **[The Emerged Security and Privacy of LLM Agent: A Survey with Case Studies](https://arxiv.org/pdf/2407.19354)** (*2024*)
  > This paper comprehensively overviews LLM agents' privacy and security issues, covers threats, impacts, defenses, trends, with case - studies to inspire future research.

- **[Inferring the Goals of Communicating Agents from Actions and Instructions](https://arxiv.org/abs/2306.16207)** (*2024*)
  > The paper models human inferential ability in cooperation. It uses GPT - 3 for instruction utterances and multi - modal Bayesian inverse planning to infer goals, showing verbal comm's importance.

- **[Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security](https://arxiv.org/abs/2401.05459)** (*2024*)
  > The paper focuses on Personal LLM Agents, discussing their architecture, challenges, and solutions, envisioning them as a major software paradigm.

- **[Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations](https://arxiv.org/abs/2410.09097)** (*2024*)
  > Paper analyzes LLM red - teaming attacks (e.g., gradient - based, RL) and defenses, aiming to foster more secure and reliable language models.

- **[Deconstructing The Ethics of Large Language Models from Long-standing Issues to New-emerging Dilemmas: A Surveyhttps://ui.adsabs.harvard.edu/](https://ui.adsabs.harvard.edu/abs/2024arXiv240605392D/abstract)** (*2024*)
  > This paper surveys LLMs' ethical challenges from old to new, analyzes related research, and emphasizes integrating ethics into LLM development.

- **[A survey on large language model based autonomous agents](https://arxiv.org/abs/2308.11432)** (*2023*)
  > This paper surveys LLM-based autonomous agents, offers a unified construction framework, overviews applications, and presents challenges and future directions.

- **[The rise and potential of large language model based agents: a survey](https://arxiv.org/abs/2309.07864)** (*2023*)
  > This paper surveys LLM-based agents, presents a general framework, explores applications, delves into agent societies, and discusses key topics and problems.

- **[Large Language Model Alignment: A Survey](https://arxiv.org/abs/2309.15025)** (*2023*)
  > This survey categorizes LLM alignment methods, explores related issues, presents benchmarks, and envisions future research for capable and safe LLMs.

- **[Ethical and social risks of harm from Language Models](https://arxiv.org/abs/2112.04359)** (*2021*)
  > Paper analyzes risks of large-scale LMs, outlines six areas with 21 risks, suggests mitigations, and points to further research directions.

- **[On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)** (*2021*)
  > This paper details opportunities and risks of foundation models, notes emergent capabilities & homogenization issues, calls for interdisciplinary research.

- **[Toward Trustworthy AI Development: Mechanisms for Supporting Verifiable Claims](https://arxiv.org/abs/2004.07213)** (*2020*)
  > The paper proposes steps for different stakeholders to enhance verifiability of AI claims, analyzes ten mechanisms, and gives related recommendations.

- **[Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commercial AI Products](https://dl.acm.org/doi/abs/10.1145/3306618.3314244?casa_token=1ogqoO70pDgAAAAA:7r8-ICJ2Ym55Fg2aaW11gpz7FR15yYHzuqBdGu7ifBfkiMRdbknxo34ItX_GwjeUZPg9k4U22tRX)** (*2019*)
  > This paper analyzes the impact of disclosing biased AI results via Gender Shades audit, showing it can prompt companies to reduce algorithmic disparities.

### Tools

- **[ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](http://arxiv.org/abs/2502.11404)** (*2025*)
  > Proposes ToolCoder, reformulating tool learning as code gen. Transforms queries to Python scaffolds, reuses code & debugs systematically.

- **[VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](https://arxiv.org/abs/2505.19255)** (*2025*)
  > Introduces VTool - R1, the first framework training VLMs for multimodal thought chains. Integrates visual tools into RFT, enabling strategic tool use without process supervision.

- **[Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval](http://arxiv.org/abs/2408.01875)** (*2024*)
  > Introduces Re-Invoke, an unsupervised tool retrieval method for large toolsets, with query synthesis, intent extraction, and multi - view ranking.

- **[Chain of Tools: Large Language Model is an Automatic Multi-tool Learner](http://arxiv.org/abs/2405.16533)** (*2024*)
  > Proposes Automatic Tool Chain (ATC) for LLMs as multi - tool users, a black - box probing method for tool learning, and builds ToolFlow benchmark.

- **[EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction](http://arxiv.org/abs/2401.06201)** (*2024*)
  > Introduces EASYTOOL, a framework that transforms diverse tool docs into concise instructions for LLMs, enhancing tool - using capabilities.

- **[ToolGen: Unified Tool Retrieval and Calling via Generation](http://arxiv.org/abs/2410.03439)** (*2024*)
  > Introduces ToolGen, integrating tool knowledge into LLM parameters via unique tokens, turning tool retrieval into generation, enhancing LLM's versatility and autonomy.

- **[ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph](http://arxiv.org/abs/2403.00839)** (*2024*)
  > The paper proposes ToolNet, a plug - and - play framework. It organizes tools into a graph, enabling LLMs to handle thousands of tools more effectively.

- **[ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback](http://arxiv.org/abs/2409.14826)** (*2024*)
  > The paper constructs MGToolBench to reflect real - world scenarios and proposes ToolPlanner with path planning and feedback for better task completion and instruction - following.

- **[Making Language Models Better Tool Learners with Execution Feedback](https://aclanthology.org/2024.naacl-long.195/)** (*2024*)
  > Proposes TRICE, a two - stage framework. Allows models to learn from tool execution feedback, deciding when and how to use tools effectively.

- **[Leveraging Large Language Models to Improve REST API Testing](https://dl.acm.org/doi/10.1145/3639476.3639769)** (*2024*)
  > The paper presents RESTGPT, leveraging LLMs to extract rules and generate values from API specs, addressing limitations of existing methods.

- **[LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error](https://aclanthology.org/2024.acl-long.570/)** (*2024*)
  > Existing LLMs have low tool-use correctness. The paper proposes STE, a bio-inspired method with trial, imagination, and memory, boosting tool learning.

- **[Skills-in-Context: Unlocking Compositionality in Large Language Models](https://aclanthology.org/2024.findings-emnlp.812/)** (*2024*)
  > The paper proposes Skills-in-Context (SKiC) prompting in in-context learning, unlocking LLMs' compositional ability and enabling zero-shot generalization.

- **[TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs](https://spj.science.org/doi/10.34133/icomputing.0063)** (*2024*)
  > Paper proposes connecting foundation models with APIs to complete tasks, leveraging models' abilities like conversation and code gen for real - world use.

- **[Gorilla: Large Language Model Connected with Massive APIs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e4c61f578ff07830f5c37378dd3ecb0d-Abstract-Conference.html)** (*2024*)
  > Developed Gorilla, a fine-tuned LLaMA, with RAT training. Mitigates hallucination and uses retrieval for better API call writing, shown via APIBench.

- **[LARGE LANGUAGE MODELS AS TOOL MAKERS](https://arxiv.org/abs/2305.17126)** (*2024*)
  > The paper presents LATM, a closed-loop framework enabling LLMs to make and use their own tools, dividing labor for cost - efficiency and extending cache applicability.

- **[Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents](http://arxiv.org/abs/2306.03314)** (*2023*)
  > A novel multi - agent framework enhances LLMs' capabilities. It addresses limitations and shows potential in AGI via diverse case - studies.

- **[Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations](http://arxiv.org/abs/2308.16505)** (*2023*)
  > Paper bridges recommender models and LLMs with "InteRecAgent", using LLMs as brain, models as tools, enabling interactive recommendation.

- **[ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](http://arxiv.org/abs/2307.16789)** (*2023*)
  > Introduces ToolLLM framework with ToolBench dataset, novel decision - tree algo, and ToolEval. Fine - tunes LLaMA to get ToolLLaMA with good generalization.

- **[TPTU-v2: Boosting Task Planning and Tool Usage of Large Language Model-based Agents in Real-world Systems](http://arxiv.org/abs/2311.11315)** (*2023*)
  > The paper introduces a framework for boosting LLM-based agents' TPTU in real-world systems, with API Retriever, LLM Finetuner, and Demo Selector.

- **[TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage](http://arxiv.org/abs/2308.03427)** (*2023*)
  > Proposes an LLM-based AI agent framework, designs two agent types, evaluates TPTU abilities, offering guidance for LLM use in AI.

- **[GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html?utm_campaign=Artificial%2BIntelligence%2BWeekly&utm_medium=email&utm_source=Artificial_Intelligence_Weekly_411)** (*2023*)
  > Proposes GPT4Tools via self - instruct to enable open - source LLMs use tools, provides a benchmark, and shows broad applicability.

- **[API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://aclanthology.org/2023.emnlp-main.187/)** (*2023*)
  > Introduces API - Bank for tool - augmented LLMs. Develops evaluation system, builds training set, and highlights future challenges.

- **[ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models](https://aclanthology.org/2023.findings-emnlp.985/)** (*2023*)
  > Proposes ChatCoT, a tool - augmented CoT reasoning framework for chat - based LLMs, models CoT as multi - turn chats, unifies reasoning and tool use.

- **[ToolQA: A Dataset for LLM Question Answering with External Tools](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9cb2a7495900f8b602cb10159246a016-Abstract-Datasets_and_Benchmarks.html)** (*2023*)
  > Introduced ToolQA dataset to evaluate LLMs' external - tool use in QA. Used scalable curation, minimized data overlap, and offered new evaluation directions.

- **[On the Tool Manipulation Capability of Open-source Large Language Models](http://arxiv.org/abs/2305.16504)** (*2023*)
  > The paper revisits classical LLM methods for open - source LLMs in tool manipulation, creates ToolBench, and offers a practical human - supervised recipe.

- **[RestGPT: Connecting Large Language Models with Real-World RESTful APIs](http://arxiv.org/abs/2306.06624)** (*2023*)
  > This paper proposes RestGPT, connecting LLMs with RESTful APIs via a planning mechanism and an API executor. It also offers RestBench for evaluation.

- **[Toolformer: Language Models Can Teach Themselves to Use Tools](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html)** (*2023*)
  > The paper proposes Toolformer, enabling LMs to self - learn using external tools via APIs with few demos, enhancing zero - shot task performance.

- **[WebCPM: Interactive Web Search for Chinese Long-form Question Answering](https://aclanthology.org/2023.acl-long.499/)** (*2023*)
  > Presents WebCPM, the first Chinese LFQA dataset with interactive web search. Records search behaviors, fine - tunes models, and makes resources public.

- **[ToolCoder: Teach Code Generation Models to use API search tools](http://arxiv.org/abs/2305.04032)** (*2023*)
  > Proposes ToolCoder, integrating API search tools into code generation. Uses ChatGPT for annotation and fine - tuning, innovatively incorporating tools in the process.

- **[ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](http://arxiv.org/abs/2306.05301)** (*2023*)
  > The paper presents ToolAlpaca, a framework to generate tool - use corpus and learn generalized skills on compact models, showing feasibility for such models.

- **[ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html)** (*2023*)
  > The paper proposes ToolkenGPT, using tool embeddings to let LLMs master tools like predicting tokens, addressing existing integration limitations.

- **[MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of Thought Prompting](https://aclanthology.org/2023.acl-short.130/)** (*2023*)
  > Proposes MultiTool - CoT, a framework using CoT prompting to integrate multiple external tools in reasoning for better performance on NumGLUE.

- **[CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models](https://aclanthology.org/2023.findings-emnlp.462/)** (*2023*)
  > Proposes CREATOR to enable LLMs to create tools, disentangling creation and execution. Introduces Creation Challenge, revolutionizing problem - solving paradigm.

- **[GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution](https://arxiv.org/pdf/2307.08775)** (*2023*)
  > Introduces GEAR, a generalizable and efficient query - tool grounding algo that delegates to SLM/LLM, improving precision at reduced cost.

- **[Dify](https://github.com/langgenius/dify)** (*2023*)
  > Dify is an open - source LLM app dev platform. Its interface integrates multiple features, enabling rapid prototype - to - production.

- **[LangChain](https://github.com/langchain-ai/langchain)** (*2023*)
  > LangChain simplifies LLM app lifecycle, offering dev components, production tools, and deployment platform for large model - based agents.

- **[WebGPT: Browser-assisted question-answering with human feedback](http://arxiv.org/abs/2112.09332)** (*2022*)
  > Fine - tune GPT - 3 for long - form Q&A with web - browsing, use imitation learning, human feedback, and reference collection, a novel approach.

- **[Task Bench: A Parameterized Benchmark for Evaluating Parallel Runtime Performance](https://www.computer.org/csdl/proceedings-article/sc/2020/999800a864/1oeOToMWZBC)** (*2020*)
  > Task Bench is a parameterized benchmark for distributed programming systems. It simplifies benchmarking and has a novel metric METG to assess systems.


## 🤝 Contributing

We welcome contributions to expand our collection. You can:
- Submit a pull request to add papers or resources
- Open an issue to suggest additional papers or resources
- Submit your paper at [our submission form](https://forms.office.com/r/sW0Zzymi5b) or email us at luo.junyu@outlook.com

We regularly update the repository to include new research.

## 📝 Citation

If you find our survey helpful, please consider citing our work:

```

@article{agentsurvey2025,
  title={Large Language Model Agent: A Survey on Methodology, Applications and Challenges},
  author={Luo, J. and Zhang, W. and Yuan, Y. and others},
  journal={arXiv preprint arXiv:2503.21460},
  year={2025}
}

```

---

<p align="center">
  <i>For questions or suggestions, please open an issue or contact the repository maintainers.</i>
</p>


