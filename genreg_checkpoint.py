# ================================================================
# GENREG Checkpoint System (2048 Edition)
# ================================================================
# Saves and loads complete population state for continued training
# ================================================================

import pickle
import os
from pathlib import Path


def save_checkpoint(population, generation, template_proteins, checkpoint_dir="checkpoints", config=None):
    """
    Save a complete checkpoint of the population state.
    Optionally saves training config so it can be restored on load.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "generation": generation,
        "population_size": population.size,
        "input_size": population.genomes[0].controller.input_size if population.genomes else None,
        "hidden_size": population.genomes[0].controller.hidden_size if population.genomes else None,
        "output_size": population.genomes[0].controller.output_size if population.genomes else None,
        "template_proteins": template_proteins,
        "config": config,
        "genomes": []
    }

    for genome in population.genomes:
        genome_data = {
            "id": genome.id,
            "trust": genome.trust,
            "max_tile": getattr(genome, 'max_tile', 0),
            "game_score": getattr(genome, 'game_score', 0),
            "step_count": getattr(genome, 'step_count', 0),
            "proteins": genome.proteins,
            "controller": {
                "input_size": genome.controller.input_size,
                "hidden_size": genome.controller.hidden_size,
                "output_size": genome.controller.output_size,
                "w1": genome.controller.w1,
                "b1": genome.controller.b1,
                "w2": genome.controller.w2,
                "b2": genome.controller.b2,
            }
        }
        # Save encoder if present (V3 evolved perception layer)
        enc = getattr(genome, 'encoder', None)
        if enc is not None:
            genome_data["encoder"] = {
                "input_dim": enc.input_dim,
                "encoder_dim": enc.encoder_dim,
                "enc_w": enc.enc_w,
                "enc_b": enc.enc_b,
                "act_id": enc.act_id,
                "act_params": enc.act_params,
                "act_bounds": enc.act_bounds,
                # Per-neuron activation params — critical for preserving
                # GPU-evolved diversity across save/load cycles.
                "act_params_per_neuron": getattr(enc, "act_params_per_neuron", None),
            }
        checkpoint_data["genomes"].append(genome_data)

    checkpoint_data["active_index"] = population.active

    checkpoint_filename = f"checkpoint_gen_{generation:05d}.pkl"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    print(f"[CHECKPOINT] Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, template_proteins=None):
    """
    Load a checkpoint and reconstruct the population.
    """
    from genreg_genome import GENREGPopulation, GENREGGenome
    from genreg_controller import GENREGController

    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    generation = checkpoint_data["generation"]
    population_size = checkpoint_data["population_size"]

    if template_proteins is None:
        template_proteins = checkpoint_data["template_proteins"]

    input_size = checkpoint_data["input_size"]
    hidden_size = checkpoint_data["hidden_size"]
    output_size = checkpoint_data["output_size"]

    import numpy as np

    genomes = []
    for genome_data in checkpoint_data["genomes"]:
        controller = GENREGController(input_size, hidden_size, output_size)
        # Auto-convert old list-based weights to numpy arrays
        controller.w1 = np.asarray(genome_data["controller"]["w1"], dtype=np.float32)
        controller.b1 = np.asarray(genome_data["controller"]["b1"], dtype=np.float32)
        controller.w2 = np.asarray(genome_data["controller"]["w2"], dtype=np.float32)
        controller.b2 = np.asarray(genome_data["controller"]["b2"], dtype=np.float32)

        genome = GENREGGenome(
            proteins=genome_data["proteins"],
            controller=controller
        )
        genome.id = genome_data["id"]
        genome.trust = genome_data["trust"]
        genome.max_tile = genome_data.get("max_tile", 0)
        genome.game_score = genome_data.get("game_score", 0)
        genome.step_count = genome_data.get("step_count", 0)

        # Restore encoder if present (V3)
        if "encoder" in genome_data:
            from genreg_encoder import GENREGEncoder, ACTIVATION_CATALOG
            enc_data = genome_data["encoder"]
            enc = GENREGEncoder(enc_data["input_dim"], enc_data["encoder_dim"])
            enc.enc_w = enc_data["enc_w"]
            enc.enc_b = enc_data["enc_b"]
            enc.act_id = enc_data["act_id"]
            enc.act_params = enc_data["act_params"]
            enc.act_bounds = enc_data["act_bounds"]
            # Restore per-neuron params if saved. If not present in legacy
            # checkpoints, rebuild from the restored act_id's defaults so
            # the keys match enc.act_id (avoids key/id mismatch).
            saved_per_neuron = enc_data.get("act_params_per_neuron", None)
            if saved_per_neuron is not None:
                enc.act_params_per_neuron = saved_per_neuron
            else:
                _, defaults, _ = ACTIVATION_CATALOG[enc.act_id]
                enc.act_params_per_neuron = [dict(defaults) for _ in range(enc.encoder_dim)]
            genome.encoder = enc
        else:
            genome.encoder = None

        genomes.append(genome)

    population = GENREGPopulation(
        template_proteins=template_proteins,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        size=population_size,
        mutation_rate=0.1
    )

    population.genomes = genomes
    population.active = checkpoint_data.get("active_index", 0)

    saved_config = checkpoint_data.get("config", None)

    print(f"[CHECKPOINT] Loaded checkpoint from {checkpoint_path}")
    print(f"  Generation: {generation}")
    print(f"  Population size: {population_size}")
    print(f"  Best trust: {max(g.trust for g in genomes):.2f}")
    print(f"  Best max tile: {max(g.max_tile for g in genomes)}")
    if saved_config:
        print(f"  Config: saved with checkpoint")

    return population, generation, template_proteins, saved_config


def list_checkpoints(checkpoint_dir="checkpoints"):
    """List all available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_gen_") and filename.endswith(".pkl"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            checkpoints.append(checkpoint_path)

    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
    return checkpoints


def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Get the path to the latest checkpoint."""
    checkpoints = list_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None
