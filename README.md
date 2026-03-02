# Drone Audio Classification (Audio-Only Research Project)

This repository hosts an **ongoing research project** on **audio-based drone recognition**,
developed in the context of an academic thesis.

The central research question is whether **drone presence and characteristics**
can be inferred **using audio signals alone**, without relying on visual or RF data.

Multiple models, architectures, and training strategies will be explored and compared over time.

---

## Project Scope

The project investigates two core tasks:

- **Drone presence detection**  
  Binary classification (drone vs. no-drone)

- **Drone motor-count classification**  
  Multi-class classification for detected drones (e.g., 2 / 4 / 6 / 8 motors)

These tasks serve as a foundation for broader experimentation with:
- Different neural architectures
- Alternative audio representations
- Training and evaluation strategies

---

## Repository Philosophy

This public repository focuses on:
- Model definitions
- Training logic
- Research-oriented experimentation code

It deliberately **does not include**:
- Audio datasets or metadata
- Dataset preparation or preprocessing scripts
- Experiment logs or trained model checkpoints
- Interim reports or thesis drafts

A **final consolidated report (PDF)** will document the complete methodology,
experiments, and conclusions at the end of the project.

---

## Code Organization

Each model or approach is contained in its own subdirectory under `models/`,
together with its associated training scripts. This structure supports
clean comparison between different approaches as the project evolves.

---

## Environment

The project is implemented in **Python** using **PyTorch**.

To create the development environment:

```bash
conda env create -f environment.yml
conda activate drone_audio
```

---

## Project Status

🚧 **Work in progress**  
This repository reflects an active research effort rather than a finalized system.

---

## Author

**Anagnostopoulos Argyrios**  
Electrical & Computer Engineering  
University of Thessaly

---

## License

This project is licensed under the MIT License.
