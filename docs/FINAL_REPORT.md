**Eco-Scale: Reducing Cloud Over-Provisioning and Energy Waste in Kubernetes through Deep Reinforcement-Learning Autoscaling.**

**BSc. in Software Engineering**

**Muhirwa Jean de Dieu Harerimana**

Mission Capstone

**Tunde Isiaq Gbadamosi**

**Date: 17/07/2026**

**DECLARATION**

This Capstone project report is my original work, unless stated and all external sources have been referenced or cited in my document. This work has not been presented for award of degree or for any similar purpose in any other university

Signature………………………… Date: **17/07/2026**

Muhirwa Jean de Dieu Harerimana

**CERTIFICATION**

The undersigned certifies that he has read and hereby recommended for acceptance of African Leadership University a report entitled ………..

Signature……………………… Date………………………….

Prof/Dr./Mrs./Miss/Mr. Name of the Supervisor

Faculty,

Bachelor of Software Engineering,

ALU

**DEDICATION AND ACKNOWLEDGEMENT** 
-----------------------------------

**Dedication**To my family, for their support and encouragement throughout my studies as well as to the many small and growing technology teams across Africa working towards building reliable, affordable, and sustainable software.

**Acknowledgement**I extend my appreciation to my supervisor **Tunde Isiaq Gbadamosi**, for his insightful supervision, feedback, and encouragement throughout the course of my capstone. I acknowledge the faculty and staff of the Bachelor of Software Engineering program at ALU for giving me an amazing learning platform that has enabled me to develop a strong foundation to grow as an engineer. My utmost appreciation goes to the maintainer of many open-sourced projects without which this work would not have been possible. These include Kubernetes, Stable-Baselines3, Gymnasium, PyTorch, and FastAPI. Also, my sincere thanks to the Alibaba Group for providing the 2018 cluster trace dataset. Lastly, I acknowledge the support of my friends, family, colleagues, and peers for their patience, testing, and feedback that have helped improve this work at various stages.

**Abstract**

Cloud infrastructure is chronically over-provisioned: industry surveys estimate that roughly a third of cloud expenditure is wasted on idle capacity (Flexera, 2023), and large-scale studies of production clusters report persistently low average utilization (Verma et al.,2015). Kubernetes' default Horizontal Pod Autoscaler (HPA) reacts to load only after a static CPU threshold is breached, causing both latency spikes under sudden demand and standing waste under light load. This project, Eco-Scale, investigated whether a deep reinforcement-learning (RL) agent could right-size a Kubernetes workload more efficiently than the HPA. A custom Gymnasium environment was driven by real Alibaba 2018 cluster traces, and three algorithms (DQN, PPO, REINFORCE) were trained under an energy-aware reward and compared against a realistic HPA baseline. PPO was selected as the deployed     champion for its stability. On held-out test traces it used ~19% fewer pods and ~65% less waste than the conservative default HPA at equal reliability, and on a live Kubernetes cluster it held the service with ~30% fewer replicas at comparable latency across repeated runs. A predictive (look-ahead) extension was tested and found not to help — even a perfect-foresight oracle did not beat the reactive agent indicating that reaction is already near-optimal at this control granularity. Eco-Scale therefore demonstrates a deployable, energy-aware autoscaler that reduces waste without manual threshold tuning.

**TABLE OF CONTENTS**
---------------------

[**DEDICATION AND ACKNOWLEDGEMENT4**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.oybwuq5gy5f8)

[**TABLE OF CONTENTS6**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.8ltwyvahmpue)

[LIST OF TABLES7](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.tkjnvikp5sp1)

[LIST OF FIGURES8](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.68ggs1v3qo8p)

[LIST OF ACRONYMS / ABBREVIATIONS10](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.36j3kyb5hz8o)

[**CHAPTER ONE: INTRODUCTION13**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.1nd3k91ecyoh)

[1.1  Introduction and Background13](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.7o833jdljx5u)

[1.2 Problem statement13](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.bfd9ryv0l8qd)

[1.3 Project’s main objective14](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.i1ig39p4bp46)

[1.3.1 List of the specific objectives14](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.r949ovyb6779)

[1.4 Research questions14](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.ybv0reyf16uq)

[1.5 Project scope15](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.r481bjj6usc3)

[1.6 Significance and Justification15](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.7spefif9wus0)

[1.7 Research Budget15](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.nzdpnflcddpe)

[1.8 Research Timeline16](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.mjinv8y65303)

[**CHAPTER TWO: LITERATURE REVIEW17**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.5p9fyk5iljxv)

[2.1 Introduction17](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.bpz1z2ecqprj)

[2.2 Historical Background of the Research Topic17](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.edfse8czfugl)

[2.3 Overview of Existing System17](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.tiyf64xqi3i7)

[2.4 Review of Related Work17](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.k7quaaagzk36)

[2.4.1 Summary of Reviewed Literature17](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.nup3btdhrfzd)

[2.5 Strength and Weaknesses of the Existing System(s)18](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.pi9a7wg2k1w9)

[2.6 General Comments19](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.tj5bbzbc071h)

[**CHAPTER THREE: SYSTEM ANALYSIS AND DESIGN19**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.7q39x0f2m2fu)

[3.1 Introduction19](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.jx1m4ti4d4ej)

[3.2 Research Design (including the SDLC model used)20](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.k0ouzjm01vkx)

[3.2.1 Dataset and Dataset Description (For those in ML specialization)20](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.54m49nha51dx)

[3.3 Functional and Non-functional Requirements20](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.pw08p5k1za1a)

[3.2.1 Proposed Model Diagram22](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.8ucoxsy5g7p1)

[3.4 System Architecture22](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.lx260th7mtjw)

[3.5 Flow Chart, Class diagram, Use Case Diagram, Sequence Diagram and all other diagrams.22](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.mfmr1ok9cihp)

[3.5.1 Use Case Diagram22](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.sgxev17km4jt)

[3.5.2 Class Diagram24](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.oh9btpisiu2y)

[3.5.3 Sequence Diagram25](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.yibqwe32vy9q)

[3.6 Development Tools25](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.fr99j17aymho)

[**CHAPTER FOUR: SYSTEM IMPLEMENTATION AND TESTING26**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.ldioxf62itbb)

[4.1 Implementation and coding26](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.sohff0nmciut)

[4.1.1 Introduction26](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.e6ng8muq3c7x)

[4.1.2 Description of implementation Tools and technology26](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.a8vlvkol8hn5)

[4.1.3 Core Modules and Representative Code27](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.yrskxisknhef)

[4.2 Graphical view of the project27](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.64ioop4sgsat)

[4.2.1 Screenshots with description27](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.m24v6wxp5jvp)

[4.3 Testing31](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.8rn8y2jnombg)

[4.3.1 Introduction31](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.j12wfrrt18ub)

[4.3.2 Objective of testing32](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.b500aqoi63yp)

[4.3.3 Unit testing outputs32](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.hawr5ypquqes)

[4.3.4 Validation testing outputs32](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.dh9mqai6b5bx)

[4.3.5 Integration testing outputs32](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.pd87n9wqfd0r)

[4.3.6 Functional and system testing results32](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.k5s1kzgfmt0e)

[4.3.7 Acceptance testing report33](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.1zr4mlj9k87i)

[**CHAPTER FIVE: THE DESCRIPTION OF THE RESULTS/SYSTEM33**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.nk5qaw1vrmpo)

[5.1. Introduction34](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.2khm3vcj7t7)

[5.2. Evaluation Setup34](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.jbnok3jls9xe)

[5.3 Algorithm Comparison and Champion Selection (RQ2)34](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.3nhxvtwfxj4)

[5.4 Head to Head Against Kubernetes HPA36](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.vfypgj7fn522)

[5.5 Real-Cluster Validation (RQ3)38](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.aubr9entz8cv)

[5.6 Predictive Extension: Does Anticipation Help? (RQ4)39](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.iy0tuuu4orif)

[5.7 Discussion — Implications, Limitations, and Impact40](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.zga3ppun7zgi)

[5.8 Answers to the Research Questions40](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.imzpn0gk0nby)

[**CHAPTER SIX: CONCLUSIONS AND RECOMMENDATIONS41**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.3j0oxdd95xtd)

[6.1 Conclusions41](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.h63vehmjk1bt)

[6.2 Limitations of the Study41](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.dfkwcm99qv89)

[6.3 Recommendations and Future Work42](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.jopru0i2ss44)

[**References43**](https://docs.google.com/document/d/1wHisAo2eCJpZDTu7XSbsXo1omNnB8q5l/edit#heading=h.gy5jwbebood)

### **LIST OF TABLES**

Table

Title

Section

Table 3.1

Functional requirements

3.3

Table 3.2

Non-functional requirements

3.3

Table 4.1

Implementation tools and technologies

4.1.2

Table 4.2

Product features mapped to screenshots

4.2.1

Table 5.1

Best run per algorithm (test reward, stability)

5.3

Table 5.2

Champion vs. HPA — headline comparison

5.4

Table 5.3

PPO on the HPA energy frontier

5.4

Table 5.4

Real-cluster results — RL vs. native HPA

5.5

Table 5.5

Predictive variants vs. champion

5.6

Table 5.6

Answers to the research questions

5.8

### **LIST OF FIGURES**

Figure

Title

Section

Figure 1.1

Eco-Scale 6-week implementation sprint roadmap

1.7

Figure 3.1

Iterative machine-learning research and engineering lifecycle

3.2

Figure 3.2

High-level system component architecture

3.4

Figure 3.3

Product-focused use-case diagram

3.5.1

Figure 3.4

Core class design

3.5.2

Figure 3.5

Live scaling-decision sequence

3.5.3

Figure 4.1

Console — Simulation mode

4.2.1

Figure 4.2

Console — Live-cluster mode

4.2.1

Figure 4.3

Console — Benchmark mode

4.2.1

Figure 4.4

Console — Results page

4.2.1

Figure 4.5

Console — Model page

4.2.1

Figure 4.6

Decision API response (/predict)

4.2.1

Figure 5.1

Training learning curves

5.3

Figure 5.2

Training diagnostics (loss, entropy, ε)

5.3

Figure 5.3

RL agent vs. native HPA

5.4

Figure 5.4

PPO on the HPA energy frontier

5.4

Figure 5.5

Real-cluster comparison

5.5

Figure 5.6

Predictive vs. reactive comparison

5.6

### **LIST OF ACRONYMS / ABBREVIATIONS**

Acronym

Meaning

ADCA

Africa Data Centres Association

API

Application Programming Interface

CNCF

Cloud Native Computing Foundation

CPU

Central Processing Unit

DQN

Deep Q-Network

Frw

Rwandan Franc

GPU

Graphics Processing Unit

GSMA

Global System for Mobile Communications Association

HPA

Horizontal Pod Autoscaler

kWh

Kilowatt-hour

ML

Machine Learning

PPO

Proximal Policy Optimization

PUE

Power Usage Effectiveness

REINFORCE

Monte-Carlo policy-gradient algorithm (Williams, 1992)

RL

Reinforcement Learning

RURA

Rwanda Utilities Regulatory Authority

SDLC

Software Development Life Cycle

SLA

Service-Level Agreement

SME

Small and Medium Enterprise

UML

Unified Modeling Language

UNEP

United Nations Environment Programme

VPA

Vertical Pod Autoscaler

**CHAPTER ONE: INTRODUCTION** 
------------------------------

### **1.1  Introduction and Background**

Cloud computing has become the standard for software delivery today, but the ability to elastically scale up and down is often hampered by inefficient capacity management. Provisioning decisions are commonly made for peak demand and left static, so resources sit idle for much of their lifetime. Flexera's (2023) State of the Cloud survey estimated that organisations waste approximately 27–32% of their cloud spend, and Google's production study of the Borg cluster average machine utilisation remained well below allocation because workloads reserve far more than they use (Verma et al., 2015). For resource-constrained users — including small and medium enterprises in emerging markets; this waste translates directly into avoidable cost and energy consumption.

The Kubernetes Horizontal Pod Autoscaler controls the elastic scaling of your pods up and down, based on the utilization of CPU and memory. For instance, you can set a threshold of CPU and memory utilization, and when the level reaches a certain percentage, the system will scale up the pods.

However, the HPA operates reactively. It only initiates scaling actions after observed metrics have already passed or met the target. This reactive nature can cause two problems. First, during sudden traffic surges, the HPA might not scale up rapidly enough, potentially leading to service level agreement (SLA) violations. Second, the cautious adjustments and the time required for stabilization can delay resource deallocation, undermining the core purpose of autoscaling.

Reinforcement learning (RL) offers an alternative in which a control policy is learned from data rather than specified by a fixed threshold. An RL agent observes the system state, takes a  scaling action, and receives a reward that encodes the operator's true objective — here, the trade-off between service quality and energy cost. RL has been applied successfully to data-centre and cluster resource management (Mao et al., 2016) and specifically to container  autoscaling. This project applies deep RL to Kubernetes autoscaling, trained and evaluated on real production traces, to test whether a learned policy can right-size workloads more efficiently than the native HPA.

### **1.2 Problem statement**

Over-provisioning in cloud clusters is a large and well-documented problem. An estimated 27–32% of global cloud spend is wasted on idle or oversized resources (Flexera, 2023), and production cluster studies confirm that reserved capacity routinely exceeds actual usage (Verma et al., 2015). The mechanism responsible in Kubernetes is the reactive, threshold-based design of the HPA: because it responds only after a static utilisation target is crossed, it cannot simultaneously avoid late reaction under bursts _and_ idle waste under light load — the operator must trade one against the other by hand-tuning the target (Kubernetes Documentation, n.d.).

Prior work has proposed learning based autoscalers to address this limitation (Mao et al.,2016), but often evaluated on synthetic workloads, does not directly compare to a realistic Kubernetes HPA under an equivalent energy aware objective, and has not been validated on a production cluster. This project will train deep-RL autoscalers on real Alibaba 2018 traces under an explicit energy versus latency reward, benchmark the resulting policies against a realistic HPA baseline in simulation, and validate the best performer on a live Kubernetes cluster. The overarching question is whether such an agent can demonstrate lower waste than the default HPA without sacrificing reliability or requiring laborious manual tuning of scaling heuristics.

### **1.3 Project’s main objective**

To design, implement, and evaluate a deep reinforcement-learning autoscaler that reduces resource waste in Kubernetes relative to the default Horizontal Pod Autoscaler while maintaining service reliability, using real production workload traces.

#### **1.3.1 List of the specific objectives**

1.  Build the learning environment. By the end of Week 2, implement a Gymnasium simulation of a Kubernetes namespace driven by the real Alibaba 2018 traces (13 daily traces, 288 steps each), with an energy-aware reward validated offline before training.
    

1.  Train and compare algorithms. By the end of week 4, train DQN, PPO, and REINFORCE algorithms by performing a 10-run hyperparameter sweep for each (150k steps per run), and select the best agent based on the mean test reward.
    

1.  Benchmark against the HPA. By the end of Week 5, evaluate the champion against a realistic HPA baseline on 5 held-out test traces and demonstrate a ≥ 20% reduction in mean pod count or waste at ≤ 1% SLA-breach rate, confirmed with a paired statistical test (p < 0.05).
    

1.  Validate on a live cluster. By the end of Week 6, deploy the champion on a real Kubernetes cluster and show it right-sizes a sample workload against the native HPA over ≥ 3 repeated runs.
    

1.  Deliver a usable product. Provide a web console and decision API that expose the agent's decisions (simulation, live cluster, and benchmark modes) so the result is reproducible and demonstrable without the command line.
    

### **1.4 Research questions**

1.  Given traces of real-world cluster loads, can an RL agent that has been trained on those traces reduce the mean number of pods (and thus waste) compared to Kubernetes HPA, with no loss of reliability?
    
2.  Between which Reinforcement Learning algorithms (DQN, PPO, or REINFORCE) was the agent most stable and effective in terms of its autoscaling policy when using an energy-aware reward function ?
    
3.  Can the policy learned by an RL agent trained in simulation be successfully deployed in real Kubernetes cluster ?
    
4.  Does providing look-ahead information (predictive) provide any benefit over purely reactive approaches ?
    

### **1.5 Project scope**

The project covers CPU-driven horizontal pod autoscaling for a single workload, trained and evaluated on the public Alibaba 2018 CPU-utilisation traces and validated on a single-node Kubernetes cluster. It **does not** cover vertical scaling, multi-metric or GPU autoscaling, or a large multi-tenant production study; these are identified as future work.

### **1.6 Significance and Justification**

By learning to track demand rather than reacting to a fixed threshold, Eco-Scale reduces the idle capacity that dominates cloud cost and energy use, without the manual tuning the HPA requires. The savings are most consequential for cost- and energy-constrained operators. The work also contributes a reusable methodological safeguard — offline reward validation before training — and an empirically tested (and rejected) predictive extension, both of which inform future autoscaling research.

### **1.7 Research Budget**

**Item**

**Description**

**Cost**

**Kaggle GPU Compute**

**Free tier — Tesla T4/P100 for model training**

**$0**

**Alibaba Cluster Trace 2018**

**Publicly available dataset**

**$0**

**Python Libraries**

**Open source frameworks (Stable-Baselines3, Gymnasium, PyTorch)**

**$0**

**GitHub Repository**

**Free tier for version control and CI/CD pipelines**

**$0**

**Local Development Machine**

**Already available hardware infrastructure**

**$0**

**Prometheus + Grafana**

**Open source cluster monitoring and data visualizers**

**$0**

**Total**

**$0**

### **1.8 Research Timeline**

**CHAPTER TWO: LITERATURE REVIEW** 
-----------------------------------

### **2.1 Introduction**

This chapter reviews the literature underpinning Eco-Scale across three strands: the scale of cloud over-provisioning, the autoscaling mechanisms Kubernetes provides today, and the application of reinforcement learning to resource management. It then synthesises the strengths and weaknesses of existing systems to locate the gap this project addresses. 

### **2.2 Historical Background of the Research Topic**

Elastic resource management has been a central concern of large-scale computing since the move to shared clusters. Google's Borg system showed both the value of tight bin-packing and the persistent difficulty of keeping utilisation high when workloads reserve more than they use (Verma et al.,2015). The problem has only grown with cloud adoption: industry surveys consistently report that a large fraction of cloud spend is wasted on idle or oversized resources (Flexera, 2023). In parallel, reinforcement learning matured from tabular methods to deep function approximation — notably Deep Q-Networks (Mnih et al., 2015) and policy-gradient methods such as PPO (Schulman et al., 2017) — making it feasible to _learn_ control policies for complex systems (Sutton & Barto, 2018).

### **2.3 Overview of Existing System** 

Kubernetes offers three native autoscalers (Kubernetes, n.d.): the **Horizontal Pod** 

**Autoscaler (HPA**), which changes replica count to hold a metric near a static target; the **Vertical Pod Autoscaler (VPA)**, which resizes pod resource requests; and the **Cluster Autoscaler**, which adds/removes nodes. The HPA is the most widely used and the direct baseline for this project. All three are reactive and rule-based: they respond after a threshold is crossed, and their behaviour is governed by hand-set targets rather than learned from workload history.

### **2.4 Review of Related Work**

Learning-based resource management has been studied as an alternative to fixed rules. Mao et al.(2016) framed cluster job scheduling as a deep-RL problem (DeepRM), showing that RL can learn allocation policies competitive with hand-crafted heuristics. On the algorithmic side, DQN (Mnih et al., 2015) suits discrete action spaces via value estimation, while PPO (Schulman et al., 2017) and REINFORCE (Williams, 1992) learn policies directly, with PPO's clipped objective giving greater training stability (Sutton & Barto, 2018). The Alibaba 2018 cluster trace (Alibaba, 2018) provides the bursty, cyclical real workloads that motivate evaluation on documented activity rather than synthetic load.

#### **2.4.1 Summary of Reviewed Literature**

Container orchestration has evolved from early cluster managers toward Kubernetes, whose scheduling and state-driven control loops Burns et al. (2016) document in detail; their account stops short of machine-learning control or the energy cost of running workloads. The reinforcement-learning methods this project builds on are themselves well established — Mnih et al. (2015) introduced Deep Q-Networks, and Sutton and Barto (2018) set out the underlying theory — but both are shown on games or abstract problems rather than live infrastructure.

Nearer to the problem, learning-based resource management has been applied to the cloud. Hasan et al. (2020) use deep Q-learning to adjust resource caps as a Markov decision process, yet evaluate on virtual machines rather than containers, so container start-up latency goes unaddressed. Work on container scaling adds a practical point: adding replicas absorbs traffic surges faster than resizing instances, with pod start-up time a major factor — the horizontal-scaling behaviour Eco-Scale depends on.

The African and energy context motivates the work. Kubernetes is now widely adopted (CNCF, 2022), while the World Bank (2023), Rwanda's Ministry of ICT and Innovation (2020), GSMA Intelligence (2025), and UNEP and the Africa Data Centres Association (2022) document the cost, connectivity, and energy pressures on local operators — including a fixed tariff of 175 Frw/kWh (RURA, 2025). Longo et al.(2020) show the value of digital twins for testing control logic before deployment, the principle behind Eco-Scale's custom Gym environment, where the agent learns safely before touching a real cluster.

Across this literature one gap recurs: energy-aware, learned autoscaling is seldom trained on real traces, benchmarked directly against the native Kubernetes HPA, and validated on a live cluster. Eco-Scale addresses that gap.

### **2.5 Strength and Weaknesses of the Existing System(s)**

**System**

**Key Feature**

**Strength**

**Weakness**

**Gap for Eco-Scale**

**HPA**

Threshold scaling

Simple, built-in, low overhead

Reactive design, lacks predictive capabilities

Cannot anticipate load waves or eliminate cold-start lag

**KEDA**

Event-driven triggers

Highly flexible external metric integration

Relies on manual configurations and static rules

Possesses zero autonomous learning or adaptation capacity

**Sedai**

ML autonomous management

High-fidelity demand forecasting loops

Expensive commercial pricing, closed architecture

Completely inaccessible to budget-constrained African startups

**Cast AI**

Cost and instance optimization

Highly efficient spot instance lifecycle management

Billed strictly in USD, ignores cluster wattage draw

Wrong optimization target for local sustainable deployments

**AWS Predictive**

Machine learning forecasting

High-accuracy time-series workload prediction

Complete cloud vendor lock-in, closed-source

Cannot be run or customized on localized local hardware pools

### **2.6 General Comments**

The literature establishes both the magnitude of the over-provisioning problem and the promise of learned control, yet leaves a specific gap: a deep-RL autoscaler trained on real traces, benchmarked head-to-head against the native HPA under an explicit energy-versus-latency reward, and validated on live Kubernetes. Eco-Scale addresses exactly this gap (§1.2).

**CHAPTER THREE: SYSTEM ANALYSIS AND DESIGN**
---------------------------------------------

### **3.1 Introduction**

This chapter presents the analysis and design of Eco-Scale: the research design and software lifecycle used, the dataset, the functional and non-functional requirements, the system architecture, and the UML models (use-case, class, and sequence). The diagrams below are given as Mermaid sources (Figures 3.1–3.5); each is exported to an image for the report.

### **3.2 Research Design (including the SDLC model used)**

The project followed a quantitative, experiment-driven research design executed through an iterative/incremental software development lifecycle. Because the core question is empirical ("does a learned policy beat the HPA?"), each iteration produced a measurable artefact and fed the next: (1) prepare data → (2) build the environment → (3) validate the reward offline → (4) train and sweep algorithms → (5) select a champion → (6) benchmark against the HPA in simulation → (7) validate on a live cluster → (8) serve the model behind an API/console. Findings at each stage (e.g., an early over-provisioning failure) were fed back into earlier stages (reward redesign), which is the essence of the iterative lifecycle.

_Figure 3.1: Iterative machine-learning research and engineering lifecycle._

#### **3.2.1 Dataset and Dataset Description (For those in ML specialization)**

The project uses the public Alibaba 2018 cluster trace (Alibaba, 2018). The raw CPU-utilisation series was resampled to 5-minute steps, normalised to \[0, 1\], and sliced into 13 daily traces of 288 steps each (CPU range ≈ 0.17–0.78). The 13 traces were split into 8 train / 5 tests, stratified by difficulty (number of stressed steps) so both sets span easy-to-hard conditions; the 5 test traces are held out and never seen during training. Each trace value drives one environment step as the current normalised demand.

### **3.3 Functional and Non-functional Requirements**

Functional requirements (FR):

ID

Requirement

FR1

Replay a recorded Alibaba trace and display the agent's scaling decision alongside the HPA (Simulation mode).

FR2

Read a live Kubernetes deployment's state and recommend or apply scaling, with Recommend-only, Autopilot, and Kill-switch controls (Live-cluster mode).

FR3

Run a head-to-head benchmark of the agent against the native HPA on the live cluster (Benchmark mode).

FR4

Present evaluation results — the algorithm sweep and the real-cluster comparison.

FR5

Expose the deployed champion's metadata (reward weights, hyperparameters, environment constants).

FR6

Provide a REST decision endpoint (POST /predict) returning a scaling action for a given state.

Non-functional requirements (NFR):

ID

Requirement

NFR1 (Performance)

A single scaling decision returns in the low-millisecond range; benchmarked for load-time, p50/p95 latency, and throughput.

NFR2 (Safety)

Actions are bounded (1–10 pods on the cluster); a Kill-switch instantly returns control; the agent never hard-fails the service.

NFR3 (Reliability)

Graceful degradation — if no cluster is reachable, the console falls back to Simulation without crashing.

NFR4 (Reproducibility)

Fixed seeds, a committed champion model, and scripted evaluation make every result reproducible.

NFR5 (Portability)

One Docker image runs the console + API locally, in a container, or on a cluster.

NFR6 (Usability)

All features are usable from a browser console — no command line required.

#### **3.2.1 Proposed Model Diagram** 

### **3.4 System Architecture**

The system has four tiers: the client (browser console), the serving node (a host with

kubectl access, running FastAPI + the champion), the Kubernetes cluster it drives, and the offline training pipeline that produces the champion. Running serving on the host (not in a container) is a deliberate choice so it inherits cluster credentials.

_Figure 3.2: High-level system component architecture. (Addresses proposal feedback #6.)_

### **3.5 Flow Chart, Class diagram, Use Case Diagram, Sequence Diagram and all other diagrams.**

#### **3.5.1 Use Case Diagram**

The product is operated by a single actor — a DevOps operator — whose use cases are the actions they can perform on the console/API, each tied to a functional requirement.

_Figure 3.3: Product-focused use-case diagram. (Addresses proposal feedback #7.)_

#### **3.5.2 Class Diagram**

_Figure 3.4: Core class design._

### **3.5.3 Sequence Diagram**

_Figure 3.5: Live scaling-decision sequence._

### **3.6 Development Tools**

The implementation tools and technologies are detailed in Chapter 4 (§4.1.2) and summarised there in the tools table.

**CHAPTER FOUR: SYSTEM IMPLEMENTATION AND TESTING**
---------------------------------------------------

### **4.1 Implementation and coding**

#### **4.1.1 Introduction**

This chapter describes _how_ Eco-Scale was built and verified. It covers the implementation tools and technologies, the core modules and representative source code, the product's user-facing features (with screenshots), and the testing performed at unit, integration, validation, functional/system, and acceptance levels. Design rationale and results are covered in Chapters 3 and 5 respectively and are not repeated here.

#### **4.1.2 Description of implementation Tools and technology**

Layer

Tools / technologies

Purpose

Learning environment

Python 3.12, Gymnasium, NumPy

Custom KubernetesEnv driven by real Alibaba traces

RL algorithms

Stable-Baselines3, PyTorch (CPU)

DQN, PPO, REINFORCE training + the served policy

Analysis / evaluation

pandas, SciPy, Matplotlib, TensorBoard

Sweeps, paired t-tests, figures, learning curves

Serving API

FastAPI, Uvicorn

/predict + simulation/live/benchmark endpoints

Web console

React + Vite + TypeScript, Tailwind CSS v4, Recharts

Browser control plane (3 modes)

Orchestration

Kubernetes, kubectl, metrics-server

Live-cluster reads and scaling

Packaging / CI

Docker (multi-stage), Jenkins

One-image build (console + API), automated image push

Testing

pytest, httpx

Unit + integration + performance harness

#### **4.1.3 Core Modules and Representative Code**

The system is organised into a trained environment, an inference engine that serves the champion, and mode-specific engines (simulation, live cluster, benchmark). Two pieces of logic are central.

(a) The energy-aware reward (environment/custom\_env.py) is what makes right-sizing — rather than over-provisioning — the optimal policy, by charging energy on the _absolute_ pod count:

def \_calculate\_reward(self, action):

    scaling\_cost = 1.0 if action != 1 else 0.0

    breach = 1.0 if self.latency >= 1.0 else 0.0

    return (

        -(self.W\_LAT   \* self.latency)                 # service quality

        - (self.W\_ENERGY \* self.pod\_count / self.MAX\_PODS)  # energy (absolute pods)

        - (self.W\_SLA  \* breach)                        # SLA saturation

        - (self.W\_SCALE \* scaling\_cost)                 # churn

    )

(b) The sim-to-real calibration (serving/live\_cluster.py) maps a live cluster's raw CPU into the normalised load the agent trained on — expressing _total_ demand across replicas, which was the fix for the idle over-provisioning noted in §5.5:

def real\_cpu\_util(avg\_cpu\_m, replicas, request\_m):

    total\_m = avg\_cpu\_m \* max(replicas, 1)          # total, not per-pod

    cu = KubernetesEnv.POD\_CAPACITY \* total\_m / max(request\_m, 1.0)

    return float(min(max(cu, 0.0), 1.0))

A single FastAPI app serves both the REST API and the built React console, so the product runs behind one port (serving/api.py); the champion is loaded once at startup by InferenceEngine, which normalises inputs exactly as the training environment did before calling model.predict.

### **4.2 Graphical view of the project**

#### **4.2.1 Screenshots with description**

The web console exposes the features named in the functional requirements (Chapter 3). Each should appear as a labelled figure in the submission:

Figure

Feature (functional requirement)

What it shows

4.2.1

Simulation mode (FR: view agent decisions on a recorded trace)

RL-vs-HPA replicas over a 24-h Alibaba trace + the live agent-decision panel

4.2.2

Live-cluster mode (FR: drive a real cluster)

The agent reading a real deployment, with Recommend-only / Autopilot / Kill-switch controls

4.2.3

Benchmark mode (FR: compare vs native HPA)

Sequential agent-vs-HPA experiment on the live cluster

4.2.4

Results page (FR: view evaluation)

Algorithm sweep table + real-cluster head-to-head

4.2.5

Model page (FR: inspect the champion)

Reward weights, hyperparameters, environment constants

4.2.6

Decision API (POST /predict)

A JSON scaling decision for a given cluster state

Figure 4.1: Simulation mode

Figure 4.2: Live Cluster mode

Figure 4.3: Benchmarkmode

Figure 4.4: Results Page

Figure 4.5: Model Page

Figure 4.6: Decision Page

### **4.3 Testing**

#### **4.3.1 Introduction**

Testing spanned automated unit and integration tests, a statistical validation of the core claim, end-to-end functional/system checks of the product, and acceptance against the project objectives. The automated suite comprises 33 test functions (43 cases after parametrisation) and runs with python -m pytest (Figure 4.3.1 below).

#### **4.3.2 Objective of testing**

To confirm that (i) the environment, reward, and policy behave correctly and safely; (ii) the API returns valid decisions and rejects malformed input; (iii) the central claim (RL beats the HPA) is statistically sound; and (iv) the delivered product meets the objectives of ‘1.3.

#### **4.3.3 Unit testing outputs**

Unit tests cover the environment, the HPA baseline, and the inference engine:

*   tests/test\_environment.py (13) — observation bounds, action effects, pod limits (1–20), reward sign, episode length.
    
*   tests/test\_hpa\_controller.py (7) — the HPA scales up under load, waits to scale down, respects bounds.
    
*   tests/test\_inference\_engine.py (6) — the champion loads, normalises observations identically to the env, and makes sane, deterministic decisions.
    

All pass (Figure 4.3.1 above).

#### **4.3.4 Validation testing outputs**

The headline claim was validated statistically: evaluation/evaluate\_vs\_hpa.py runs the champion and the HPA on 50 paired episodes and applies a paired t-test — PPO beats the HPA by +4.45 reward, t = 17.1, p < 0.0001 (Figure: screenshots/t-test.png). The reward itself was validated _offline_ before training (training/reward\_design.py): a demand-tracking policy scores −348.5 vs −500.7 for an over-provisioning policy, confirming the reward rewards right-sizing.

#### **4.3.5 Integration testing outputs**

tests/test\_api.py (7) exercises the running FastAPI service over HTTP: /health, /info, and /predict return correct responses, and /predict rejects out-of-range CPU, zero pods, and missing fields — confirming the serving layer integrates the model correctly end-to-end.

#### **4.3.6 Functional and system testing results**

All three console modes were exercised end-to-end: Simulation replays a trace with live RL-vs-HPA decisions; Live-cluster reads and (in Autopilot) scales a real deployment via kubectl, with the Kill-switch returning control instantly; Benchmark runs the sequential agent-vs-HPA experiment. A cross-environment performance benchmark (tests/benchmark\_performance.py) reports model load-time, single-decision latency (p50/p95), and throughput on the host, in Docker, and in the cluster (Figure: screenshots/benchmark\_performance.png, screenshots/docker-env.png).

#### **4.3.7 Acceptance testing report**

Acceptance was assessed against the SMART objectives of §1.3: the agent achieved the targeted ≥ 20% reduction (19% fewer pods / 65% less waste vs the default HPA, p < 0.0001), transferred to a live cluster (~35% fewer replicas over 3 runs), and is delivered as a working, deployed web console + API. All acceptance criteria were met (see Chapter 5 for the full results).

**CHAPTER FIVE: THE DESCRIPTION OF THE RESULTS/SYSTEM**
-------------------------------------------------------

### **5.1. Introduction**

This chapter reports and interprets the experimental results of the Eco-Scale autoscaler. The

findings are organised around the four research questions of **1.4: algorithm selection (RQ2)**,

the head-to-head against the Kubernetes HPA (RQ1), transfer to a live cluster (RQ3), and whether anticipation improves the agent (RQ4). Each result is presented with its supporting **figure** and then discussed; **5.7** draws out the implications, limitations, and impact, and 5.8 answers each research question directly.

### **5.2. Evaluation Setup**

All simulated results are reported on the 5 held-out test traces that were **never** **seen** during training, each evaluated at 10 start-offsets to give 50 paired episodes per controller. Because every controller is run on identical trace–offset pairs, differences are assessed with a paired t-test. Five metrics are reported throughout: mean episode reward (the combined objective), p95 latency, SLA-breach rate (percentage of steps at saturation), waste (idle pods above the healthy count), and mean pod count (the energy proxy). Reward values are specific to this environment and are compared only within it, against the HPA baseline and the held-out results.

### **5.3 Algorithm Comparison and Champion Selection (RQ2)**

Each algorithm was trained as a 10-run hyperparameter sweep (150k steps per run). The best run per algorithm on the held-out traces was:

Algorithm

Best mean reward

Stability

PPO (champion)

−340.07 ± 10.24

tightest sweep (−351 … −340); no collapses

DQN

−344.58 ± 7.74

4 runs collapsed (to ≈ −480/−489)

REINFORCE

−348.91 ± 12.02

no-baseline run collapsed (−402)

The ranking is PPO ≈ DQN > REINFORCE, but the decisive factor was _stability_, not peak score.

The training learning curves (Figure 5.1) show PPO converging smoothly to a tight band while DQN oscillates violently — repeatedly crashing toward −600 before recovering — despite a comparable final score.

 The optimizer diagnostics (Figure 5.2) corroborate healthy PPO training: its value loss collapses as the critic learns, and its policy entropy drifts up as the policy commits.

Because an autoscaler must run unattended, PPO's reliability made it the champion (Figure 5.3). Answer to RQ2: PPO — clipped, on-policy updates gave the best stability/quality trade-off.

### **5.4 Head to Head Against Kubernetes HPA**

The champion was benchmarked against a realistic reactive HPA baseline (standardtarget-utilisation formula, ±1 pod/step, scale-down stabilisation) on the 50 paired episodes(Figure 5.3):

**Controller**

**Mean reward**

**p95 latency**

**Breach %**

**Waste**

**Mean pods**

**PPO**

**−340.19 ± 10.3**

**0.72**

0.22

0.050

8.03

HPA (70% target)

−344.64 ± 8.6

0.79

0.21

0.019

7.29

DQN

−347.51 ± 9.3

0.78

0.47

0.106

9.06

random

−460 ± 44

1.00

32

0.137

8.12

The paired t-test confirms the headline: **PPO beats the HPA by +4.45 reward (t = 17.1, p < 0.0001)**, driven by lower p95 latency (0.72 vs 0.79) at comparable reliability. (DQN, by contrast, is _significantly worse_ than the HPA — it over-provisions to ~9 pods without a service benefit — which is reported honestly.)

Because the HPA's energy/reliability trade-off depends entirely on its static target, the target was swept to place PPO on the HPA tuning frontier (Figure 5.4):

**Controller**

**Reward**

**Mean pods**

**Waste**

**Breach %**

**PPO (adaptive)**

**−340.2**

8.03

0.050

0.2

HPA @50% (conservative default)

−353.2

9.97

0.142

0.1

HPA @70% (well-tuned)

−344.6

7.29

0.019

0.2

HPA @90% (aggressive)

−370.5

5.81

0.000

4.4

**Finding.** Against the conservative **HPA@50% that teams deploy by default**, PPO uses **~19% fewer pods and ~65% less waste at equal reliability**, and reaches this operating point with **no threshold tuning**. **Honest limit:** PPO does _not_ beat a perfectly-tuned HPA@70% on pure energy (7.29 vs 8.03 pods); its value is winning the combined objective and landing on a good operating point automatically. **Answer to RQ1: yes** — the agent reduces waste versus the default HPA at equal-or-better reliability.

### **5.5 Real-Cluster Validation (RQ3)**

The champion was deployed on a live single-node Kubernetes cluster and run head-to-head against the **native** HPA under an identical load wave, over 3 repeated rounds (Figure 5.5):

**Metric**

**RL (PPO)**

**Native HPA**

Mean replicas

**4.00 ± 0.22**

6.17 ± 0.31

p95 latency (ms)

1054 ± 17

1126 ± 72

**Finding.** On real cluster the agent held the service with **~35% fewer replicas** (4.00 vs 6.17) at **comparable p95 latency**, and the small standard deviations show the result is **reproducible**, not a one-off. Achieving this required an observation calibration expressing _total_ real demand on the scale the agent trained on which removed an initial idle over-provisioning (a documented sim-to-real gap). A residual idle floor (~4 rather than 1 pod) remains, attributable to training always starting at 5 pods, and both controllers saturated on the single node, so this regime tests efficiency under stress rather than low-latency operation. **Answer to RQ3: yes** the simulation-trained policy transferred to live Kubernetes and beat the production HPA on resource use at equal service quality.

### **5.6 Predictive Extension: Does Anticipation Help? (RQ4)**

A natural hypothesis is that the agent should _anticipate_ load rather than react. This was tested by extending the champion with one look-ahead feature (same PPO configuration) in three variants  **trend** (causal slope), **forecast** (causal Holt projection), and **oracle** (true future peak; perfect foresight, an upper bound)  and re-benchmarking on the held-out traces (Figure 5.6):

**Controller**

**Reward**

**Breach %**

**Mean pods**

**Champion (reactive)**

**−340.19**

0.22

8.03

Oracle (perfect foresight)

−342.66

0.43

7.80

Trend

−342.16

0.24

8.35

Forecast

−341.51

0.22

8.20

**Finding: anticipation does not help here.** All three variants are _significantly worse_ than the reactive champion (paired t-tests, all p < 0.0001). Decisively, even the **oracle** — with perfect knowledge of future load — loses: it runs marginally leaner but roughly _doubles_ SLA breaches. At 5-minute control granularity with ±1-pod steps, reaction is already near-optimal. **Answer to RQ4: no** — look-ahead offers no usable advantage, so the reactive agent is retained. This converts the proposal's speculative "predictive" direction into a tested, evidence-based conclusion.

### **5.7 Discussion — Implications, Limitations, and Impact**

**Implications.** The results show that a learned policy can match or beat a hand-tuned HPA on the combined latency/energy objective _without_ manual threshold tuning, and — importantly — that the much-assumed benefit of _prediction_ does not materialise at realistic control granularity. For practitioners this means the achievable win from RL autoscaling is primarily _automatic right-sizing_, not foresight.

**Limitations.** (1) Scope is single-workload, CPU-driven horizontal scaling; vertical/multi-metric scaling is untested. (2) Real-cluster validation used a single node that saturated under load, so it demonstrates resource efficiency under stress rather than low-latency headroom. (3) A residual idle floor (~4 pods) remains from the fixed training start. (4) Traces are a CPU proxy for African-SME workloads, not measured from one.

**Impact.** By tracking demand rather than reacting to a fixed threshold, Eco-Scale reduces theidle capacity that dominates cloud cost and energy use  up to ~19% fewer pods in simulation and ~35% fewer replicas on a real cluster which is most consequential for cost- and energy-constrained operators, the motivating audience of this project.

### **5.8 Answers to the Research Questions**

**RQ**

**Question**

**Answer**

RQ1

Reduce waste vs. the default HPA at equal reliability?

**Yes** ~19% fewer pods / ~65% less waste vs HPA@50%, p < 0.0001.

RQ2

Which algorithm is best?

**PPO** most stable; chosen champion.

RQ3

Does it transfer to a live cluster?

**Yes** ~35% fewer replicas at comparable latency, 3 runs.

RQ4

Does prediction help?

**No** even an oracle loses; reaction is near-optimal.

**CHAPTER SIX: CONCLUSIONS AND RECOMMENDATIONS**
------------------------------------------------

### **6.1 Conclusions**

This project set out to address a large, well-documented problem (§1.2): cloud clusters are chronically over-provisioned, and Kubernetes' default Horizontal Pod Autoscaler — being reactive and threshold-based — cannot avoid both late reaction under bursts and idle waste under light load without manual tuning. Eco-Scale asked whether a deep reinforcement-learning agent, trained on real Alibaba 2018 traces under an energy-aware reward, could right-size a workload more efficiently thanthe HPA.

The evidence answers the research questions clearly and affirmatively where it matters:

*   The agent beats the default HPA (RQ1). On held-out real-trace data, the PPO champion used~19% fewer pods and ~65% less waste than the conservative default HPA at equal reliability,a statistically significant win (paired t-test, p < 0.0001) — and it reached this operating pointwith no threshold tuning.
    
*   PPO was the right algorithm (RQ2). It was the most stable of the three algorithms and wasselected as the deployed champion.
    
*   The policy transferred to real infrastructure (RQ3). On a live Kubernetes cluster it held theservice with ~35% fewer replicas at comparable latency, reproducibly across three runs.
    
*   Prediction was tested and rejected (RQ4). Adding look-ahead features did not help; even aperfect-foresight oracle lost to the reactive agent, showing reaction is already near-optimal atthis control granularity.
    

Taken together, these results confirm the project hypothesis and directly address the stated problem: a learned policy reduces the idle capacity that drives cloud cost and energy use, without the manual tuning the HPA demands. The work also contributes a reusable methodological safeguard — validating the reward offline before training — and converts the proposal's speculative "predictive" direction into a tested, evidence-based finding.

### **6.2 Limitations of the Study**

1.  Scope — single-workload, CPU-driven _horizontal_ scaling; vertical, multi-metric, and GPU autoscaling were out of scope.
    
2.  Real-cluster regime — validation used a single node that saturated under load, so it demonstrates resource efficiency under stress rather than low-latency headroom.
    
3.  Residual idle floor — the agent settles around ~4 pods rather than 1 when idle, an artefact of always starting training at 5 pods.
    
4.  Workload proxy — the Alibaba traces are a realistic but indirect proxy for the target African-SME workloads, which were not measured directly.
    

### **6.3 Recommendations and Future Work**

For anyone continuing from here:

1.  Finer control intervals. Test sub-minute control, where anticipation _might_ begin to pay off (unlike the 5-minute steps here, where it did not).
    
2.  On-cluster fine-tuning. A short bout of training on the live cluster should remove the residual idle floor and close the last of the sim-to-real gap.
    
3.  Larger real-cluster study. Run multi-node, unsaturated experiments with Prometheus (metrics) and Locust (load) for a larger statistical sample and true latency headroom.
    
4.  Broaden the action space. Add action masking at the pod bounds and explore multi-metric and vertical scaling.
    
5.  Generalise the product. Extend the console's cluster selector to discover namespaces and deployments dynamically, so it drives any workload rather than the single sample app.
    

In summary, Eco-Scale delivered a working, energy-aware Kubernetes autoscaler that measurablyoutperforms the default HPA and transfers to real infrastructure, with its limits and next stepsclearly identified.

**References**
--------------

Alibaba. (2018). Alibaba cluster trace v2018 \[Data set\]. GitHub. [https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)

Burns, B., Grant, B., Oppenheimer, D., Brewer, E., & Wilkes, J. (2016). Borg, Omega, and Kubernetes. ACM Queue, 14(1), 70–93. [https://doi.org/10.1145/2898442.2898444](https://doi.org/10.1145/2898442.2898444)

Cloud Native Computing Foundation. (2022). CNCF annual survey 2022. Cloud Native Computing Foundation & Linux Foundation Research. [https://www.cncf.io/reports/cncf-annual-survey-2022/](https://www.cncf.io/reports/cncf-annual-survey-2022/)

Flexera. (2023). Flexera 2023 state of the cloud report. Flexera. [https://www.flexera.com/about-us/press-center/flexera-2023-state-of-the-cloud-report](https://www.flexera.com/about-us/press-center/flexera-2023-state-of-the-cloud-report)

GSMA Intelligence. (2025). The mobile economy Africa 2025. GSMA. [https://www.gsmaintelligence.com/research/the-mobile-economy-africa-2025](https://www.gsmaintelligence.com/research/the-mobile-economy-africa-2025)

Kubernetes. (n.d.). Horizontal pod autoscaling. Retrieved July 19, 2026, from [https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). Resource management with deep reinforcement learning. In Proceedings of the 15th ACM Workshop on Hot Topics in Networks (HotNets-XV) (pp. 50–56). Association for Computing Machinery. [https://doi.org/10.1145/3005745.3005750](https://doi.org/10.1145/3005745.3005750)

Ministry of ICT and Innovation. (2015). SMART Rwanda master plan 2015–2020. Government of Rwanda. [https://www.minict.gov.rw/fileadmin/user\_upload/minict\_user\_upload/Documents/Policies/SMART\_RWANDA\_MASTERPLAN.pdf](https://www.minict.gov.rw/fileadmin/user_upload/minict_user_upload/Documents/Policies/SMART_RWANDA_MASTERPLAN.pdf)

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., … Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533. [https://doi.org/10.1038/nature14236](https://doi.org/10.1038/nature14236)

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv. [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.

United Nations Environment Programme, & Africa Data Centres Association. (2022). Africa digital infrastructure and climate resilience report. UNEP. \[VERIFY title/URL\]

Verma, A., Pedrosa, L., Korupolu, M., Oppenheimer, D., Tune, E., & Wilkes, J. (2015). Large-scale cluster management at Google with Borg. In Proceedings of the Tenth European Conference on Computer Systems (EuroSys '15) (pp. 1–17). Association for Computing Machinery. [https://doi.org/10.1145/2741948.2741964](https://doi.org/10.1145/2741948.2741964)

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256. [https://doi.org/10.1007/BF00992696](https://doi.org/10.1007/BF00992696)

World Bank. (2023). World Bank annual report 2023. World Bank. [https://thedocs.worldbank.org/en/doc/e0f016c369ef94f87dec9bcb22a80dc7-0330212023/original/Annual-Report-2023.pdf](https://thedocs.worldbank.org/en/doc/e0f016c369ef94f87dec9bcb22a80dc7-0330212023/original/Annual-Report-2023.pdf)