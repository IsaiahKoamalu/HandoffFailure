#import "proposal.typ": proposal

#show: doc => proposal(doc,
  title: [Measuring Handoff Failure Between Independently Trained Reinforcement Learning Skills],
  authors: (
    (
      name: "Isaiah Broderson",
      affiliation: "University of Hawaiʻi at Hilo",
      email: "isaiahkb@hawaii.edu",
    ),
  ),
  abstract: [Long-horizon control problems are rarely able to be solved by a singular reinforcement learning policy. The standard approach is to train narrow skills separately and then chain their execution together in sequence, which replaces one hard credit-assignment problem with a set of easy ones. However this introduces a failure mode that is not often measured directly. Each skill is trained on its own distribution of starting states, but when chained it receives whatever state its predecessor produced. In this project I will train two spacecraft control skills independently inside an existing physics simulator, chain them into a single mission, and measure both the loss in mission success and the distributional distance between the delivered handoff state and the successor's training distribution. The central claim to be tested is that the second quantity predicts the first. The reinforcement learning implementation, including Proximal Policy Optimization, the task definitions, the policy sequencer and the evaluation harness, will be written for this project; the simulator predates it and is treated as fixed apparatus.]
)



= Introduction
Reinforcement learning works by giving credit to actions from the reward they eventually produce. That mechanism degrades as the horizon grows. For a spacecraft mission that begins in orbit around one body and ends on the surface of another spans tens of thousands of simulation steps, and a single terminal reward provides the agent with nearly no information as to which decisions mattered most. The action spaces are not consistent across the mission, moving a character and flying a spacecraft with six degrees of operational freedom are different control problems joined end to end.

The usual response to this problem is temporal abstraction. Rather than learning a whole mission, it is decomposed into skills, where each skill is trained against a dense local reward, and then executed in sequence. Each skill then faces a controlled problem, and the resulting system can be inspected one component at a time.

This decomposition carries an assumption that is rarely measured. Each skill is trained on a distribution of starting states chosen by the designer. When the skills are chained a skill no longer receives a sampled starting state, it receives whatever state the preceding skill happened to leave the system in. If those two distributions disagree then a skill that is individually reliable can fail in composition without anything being wrong with the skill itself. So the failure therefore belongs to the 'seam', where the two skills meet, not to either component itself.

This project treats that seam as the object of study. I will train two control skills independently, chain them, and pose two questions. First, how much mission success is lost relative to what the individual success rates predict. Second, whether that loss is explained by a direct measurement of how far the delivered handoff state falls outside the successor's training distribution, as shown in @fig-chain.
#figure(
  image("chain.png"),
  caption: [The mission as a chain of skills. The seam studied here
  is the one between orbit insertion and landing.],
) <fig-chain>



= Prior Work
Temporal abstraction is most often formalized through the options framework of Sutton, Precup, and Singh @sutton1999, which takes the standard reinforcement learning setup and lets entire multi-step skills be represented as single actions, called options, each one possessing a rule for when it can begin and when it stops. The initiation set of a trained skill is what describes its own valid starting states. The framework assumes that an option is only applied from inside that set. However, when skills are trained separately nothing enforces it, and that is the situation this project measures. Later work learns the decomposition rather than actually specifying it, the option-critic architecture @bacon2017 learns options as well as their terminations end to end, and Nachum et al. @nachum2018 train a higher-level policy that issues goals to a lower-level one. Both avoid the handoff problem by training the skills together. This project will deliberately keep the skills independent, since that is the common engineering practice and the case in which the seam is exposed.

The underlying issue lies in distribution shift, meaning the skill faces situations that look much different than what it was trained on. This is the same phenomenon that motivates DAgger @ross2011. There an agent learns from an expert and then eventually drifts into situations which the expert never taught it, a remedy is to collect labels on the states that the agent reaches. This structure is identical at a skill handoff, where the predecessor skill takes the role of the thing that produces unfamiliar states. Domain randomization @tobin2017 looks at the same problem but from a different angle, widening the training distribution until the deployment distribution falls inside it.

The current learning algorithm is Proximal Policy Optimization @schulman2017, which I chose for stability and for being simple enough to implement correctly from the paper. The reward shaping follows the potential-based form of Ng, Harada and Russell @ng1999, which guarantees that shaping cannot change the optimal policy. This is important because the training of each skill relies on a reward that is dense enough to train on.

What I have not found is a direct measurement of the handoff state distribution as a predictor of the chained-system failure. The literature I have found largely avoids the problem by training jointly, and the imitation literature studies distribution shift between a learner and an expert rather than between two separate learned skills.\

= Data
The data source is a simulator I built in GODOT 4; this predates the project and will be treated as a fixed apparatus. The simulator provides functioning planetary orbits, procedurally generated terrain based on a per-planet seed, and a rigid-body ship with six degrees of thrusting freedom, a finite fuel system and real gravitational dynamics. Since the ship obeys gravity while the planets are on rails, the ship's trajectory is a real orbital mechanics problem. Periapsis, apoapsis and eccentricity are all computed from position and velocity rather than scripted.

Data is generated rather than collected. An episode is produced by stepping 32 ships in parallel, which reaches roughly 5,000 environment steps per second on a laptop CPU, so a three-million-step run completes well under an hour. A single decision is held for six physics ticks, which places a full episode inside the discount horizon. Five of the planetary bodies have terrain and are able to be landed on, all differing in radius, surface gravity, and terrain seed. One is held out of training entirely, which will test whether a skill has learned to read terrain rather than memorize the bodies it has trained on.\
\


= Models
An actor-critic implementation of Proximal Policy Optimization @schulman2017 written for this project, part of it keeps a running record of what "normal" inputs look like for each skill. During training, each skill keeps a record of the typical value and spread of each number it sees, which is saved alongside the training weights. The normal purpose of this is to keep inputs at a reasonable scale. It also provides a description of what the skill considers a familiar situation, which is what I will use to measure how unusual the handoff is. There will be two skills, one in charge of getting into orbit and one in charge of descending and landing.

The orbit insertion skill takes a ship approaching a planet on an arbitrary trajectory and places it in a "parking orbit". Success is defined by the orbital elements rather than position: periapsis clear of the terrain, apoapsis inside the planet's capture shell, eccentricity below a threshold, orbital radius within tolerance of a target, and the whole condition held long enough so that a trajectory just crossing the border does not count. In short, the ship must settle into a stable, roughly circular orbit at the right height without hitting the ground or running out of fuel, and stay there, not just land.

The descent and landing skill takes a ship in orbit and lands it intact on a designated continent, scored on touchdown speed, attitude and whether the landing site is on land.

Finally there will be a mission runner that makes the composition possible. The two skills do not currently share an interface. They read different lists of values (26 against 20). Descent and landing expects the ship's auto-stabilizer to be on while insertion requires it to be off. One also flies on a fuel budget while the other does not, which means they assume different limits on how long a run may last. The mission runner is responsible for reconciling this mid flight by holding the ship's state across the transition, swapping the active observation builder and policy while reconfiguring the ship and deciding when the handover should occur.

#table(
  columns: (auto, auto, auto),
  [*Property*], [*Orbit insertion*], [*Descent and landing*],
  [Inputs read], [26 values], [20 values],
  [Auto-stabiliser], [Off], [On],
  [Fuel], [Finite budget], [Unlimited],
  [Run length limit], [18,000 ticks], [27,000 ticks],
)


Each skill is also implemented as a hand written analytic controller. These serve as a baseline and establish what the task is worth without learning, and provide a control that cannot memorize terrain.\
\


= Experiments

*E1.* I will measure how often each skill succeeds by itself, which will set the overall expectation.

*E2.* Then I will run the chained mission and compare its success rate to what the individual rates predict. For example if each skill succeeds 90% of the time you would expect about 81% together. So a performance of much worse means the handoff is costing something.

*E3.* For each of the values that the descent and landing skill reads, it has a saved record of the typical value seen during training and the typical amount of variation. At the handoff I will compare the delivered situation against that record and express each value as the number of typical variations it sits away from usual. Then I will summarize the result in two ways: the largest single departure, and how many values sit more than three variations from usual. Both will be recorded against whether the mission succeeded, and if the stranger handoffs are the ones that fail, then the central claim holds. I will then repeat the same handoff measurement at a second boundary, where the ship is handed over by the hand written transit controller rather than by a learned skill. If the relationship between unusual handoffs and failed missions appears at both boundaries, it is a general property.

*E4.* I intend to implement three fixes and see which one helps the most. Firstly I will retrain the successor over a broader range of starting states to emphasize domain randomization and a wider initial distribution so that the handoff state is no longer unusual. Second I plan to run the predecessor and collect the states that it actually produces and continue to train the successor on those. The third fix involves replacing the fixed handover criterion with a small learned policy that selects the moment of transfer.\

There are three controls. If a training curve looks surprising I will re-run it with the learning rate set to zero. A curve that reproduces with a frozen network stems from the environment rather than the learning. Any evidence that a skill is generalizing poorly will be checked against the handwritten controller on the same held out world. This serves as a baseline-drop control since the controller has nothing to memorize, so if it loses the same amount then the world is harder rather than the skill overfitting. The final control involves two unit tests, one for runs that are cut off by the clock, and a second to regulate the reward scale. A run that is stopped by the time limit must be scored differently than one that has truly ended, since scoring it normally would result in rewarding the agent for running out the clock. I will work out a small case by hand and check it against this. For the reward scale, the agent pays a small cost for each step it continues, if that cost is able to exceed the failure penalty then it will recognize that quitting on purpose is the cheapest option. I will implement an assertion at startup to make sure that a maximum length run cannot cost more than the smallest failure penalty.

Every measurement is 200 attempts, repeated across five different random settings, I will be tracking the average along with how much it varied. Controllers that are being compared will be given the identical set of starting situations rather than new random ones to ensure that a difference between them does not stem from one drawing easier cases. When scoring, the agent's exploration randomness will be switched off so that it acts on its best guess. Once each skill has been measured on its own, the rules of both tasks will be fixed (what counts as success, what is rewarded, how long a run may last). Any changes made after that point will be reported as its own comparison showing the result both before and after. This will prevent adjustment of the rules for favorable results.


= Impact
The practical contribution is a measurement procedure. If the idea holds, anyone chaining separately trained skills gets a cheap early-warning diagnostic check before deployment. Before running the whole system, compare each handoff to what the next skill already knows, and then you'll know which handoffs are too risky.

The negative result is also useful. If the composition succeeds where the individual success rates predict that it should, that is evidence that independently trained skills compose more robustly than the covariate-shift argument suggests, which narrows where the hierarchical reinforcement learning literature's emphasis on joint training is actually necessary.

More broadly, autonomous spacecraft operation is a domain where missions are naturally phased and where each phase already possesses well understood classical control laws. Knowing where learned components can be substituted into such a sequence, and what has to be verified at boundaries, is a key prerequisite for using them at all.\


= Timeline

#table(
  columns: (auto, 1fr),
  [*Week Of*], [*Description*],
  [Oct 26], [PPO implementation, environment wrapper and engine bridge (unit tests on the learning code). Landing skill reproduced at its reference rate.],
  [Nov 2], [Second skill trained. E1 reference measurements at five seeds.],
  [Nov 9], [Mission runner across the interface mismatch. Running E2 and E3 on the learned$->$learned seam.],
  [Nov 16], [Prepare progress report],
  [Nov 23], [Second seam (analytic$->$learned). Replicate E3 on this seam.],
  [Nov 30], [E4 repairs implemented and compared.],
  [Dec 7], [Buffer, analysis, figures, paper.],
  [Dec 13], [Finalize code, prepare presentation]
)


#bibliography("main.bib")
