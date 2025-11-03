# NeuroLobster

Embodied simulation of the lobster stomatogastric nervous system.

## Goal

Implement a biologically accurate neural circuit (STG of *Homarus americanus*)
and connect it to a physically simulated body. No reinforcement learning.
No reward functions. Pure biological circuit dynamics.

## Planned stack

- Brian2 — spiking neural network simulation
- MuJoCo — physics engine
- Hodgkin-Huxley — conductance-based neuron model

## Status

[ ] STG pyloric rhythm model
[ ] MuJoCo body definition
[ ] Closed sensorimotor loop

## References

- Marder & Bucher (2007). Understanding circuit dynamics using the STG. Neuron.
- Prinz, Bucher & Marder (2004). Similar network activity from disparate circuit parameters. Nature Neuroscience.
