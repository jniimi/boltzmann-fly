# Draft entry for cobanov/awesome-fly (Step 3)

The list uses `- [Name](url) by **owner** - description.` with bold tags such as **Research prototype**
inside the description. (The list's README contains fly imagery; do not open it to copy formatting,
the format above was extracted as text.) Suggested section: "Language, art, and other experiments"
(alternatively "Brain models and embodied simulation").

Entry:

- [boltzmann-fly](https://github.com/jniimi/boltzmann-fly) by **jniimi** - Energy-based world model of simulated consumer behaviour (Purchase World, ICONIP 2026) re-trained as a Boltzmann machine whose couplings are masked to the MaleCNS mushroom-body wiring (PN -> Kenyon cell -> MBON, right hemisphere); ships degree-preserving and Erdos-Renyi rewiring controls, CPU-reproducible weights, and an honest negative result. **Research prototype**

PR title: `Add boltzmann-fly (mushroom-body-masked Boltzmann machine world model)`

PR body (plain text, no images):

> Adds boltzmann-fly, a hobby port of the Purchase World energy-based world model onto the MaleCNS v1.0
> mushroom-body wiring. Only the 0/1 coupling pattern is taken from the connectome; magnitudes are
> learned. Includes the original dense model as an anchor, degree-preserving and Erdos-Renyi
> controls, unit-tested masking, and a results note. Text-only repository (no neuron renderings).
