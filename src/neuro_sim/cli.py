"""Command-line interface for neuro_sim."""

# Apply compatibility patches before importing bindsnet
import neuro_sim.compat  # noqa: F401

import argparse
import sys
from pathlib import Path

from neuro_sim.config import Config
from neuro_sim.training import Evaluator, Trainer
from neuro_sim.utils.logging import setup_logger


def train_command(args):
    """Train command."""
    # Setup logging
    log_dir = args.log_dir
    logger = setup_logger(name="neuro_sim.train", log_dir=log_dir)
    
    try:
        # Load config
        if args.config:
            config = Config(config_path=args.config)
        else:
            config = Config()
        
        # Override config with command-line arguments
        if args.model:
            config.set("model.name", args.model)
            logger.info(f"Model type overridden by CLI: {args.model}")
        if args.batch_size:
            config.set("training.batch_size", args.batch_size)
        if args.n_epochs:
            config.set("training.n_epochs", args.n_epochs)
        if args.n_train:
            config.set("training.n_train", args.n_train)
        
        # Handle GPU setting: only override if explicitly set via CLI
        # Check if --gpu or --no-gpu was explicitly provided in command line
        gpu_flag_provided = '--gpu' in sys.argv or '--no-gpu' in sys.argv
        
        if gpu_flag_provided and args.gpu is not None:
            config.set("training.gpu", args.gpu)
            logger.info(f"GPU setting overridden by CLI: {args.gpu}")
        else:
            # Use config value (from YAML or default)
            gpu_from_config = config.training.get("gpu", True)
            logger.info(f"Using GPU setting from config: {gpu_from_config}")
        
        # Optimization flags (default to disabled for speed)
        if args.enable_monitoring is not None:
            config.set("training.enable_monitoring", args.enable_monitoring)
        else:
            # Default to False if not specified
            config.set("training.enable_monitoring", config.training.get("enable_monitoring", False))
        
        if args.enable_grad_tracking is not None:
            config.set("training.enable_grad_tracking", args.enable_grad_tracking)
        else:
            # Default to False if not specified
            config.set("training.enable_grad_tracking", config.training.get("enable_grad_tracking", False))
        
        if args.enable_weight_norm is not None:
            config.set("training.enable_weight_norm", args.enable_weight_norm)
        else:
            # Default to False if not specified
            config.set("training.enable_weight_norm", config.training.get("enable_weight_norm", False))
        
        if args.checkpoint_dir:
            config.paths["checkpoint_dir"] = args.checkpoint_dir
        
        logger.info("Starting training...")
        logger.info(f"Config: {config.to_dict()}")
        if args.model_name:
            logger.info(f"Model will be saved as: {args.model_name}.pt")
        
        # Create trainer
        trainer = Trainer(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
        )
        
        # Train
        resume_from = args.resume_from
        metrics = trainer.train(resume_from=resume_from)
        
        logger.info(f"Training completed. Final metrics: {metrics}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def infer_command(args):
    """Inference command."""
    # Setup logging - use root "neuro_sim" logger so child loggers also work
    log_dir = args.log_dir
    logger = setup_logger(name="neuro_sim", log_dir=log_dir)
    
    try:
        # Load config
        if args.config:
            config = Config(config_path=args.config)
        else:
            config = Config()
        
        # Override config with command-line arguments
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = None
        
        if args.batch_size:
            config.set("inference.batch_size", args.batch_size)
        if args.gpu is not None:
            config.set("inference.gpu", args.gpu)
        
        logger.info("Starting inference...")
        logger.info(f"Using model: {model_path}")
        
        # Create evaluator
        evaluator = Evaluator(
            config=config,
            model_path=model_path,
        )
        
        # Evaluate
        metrics = evaluator.evaluate()
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


def train():
    """Train entry point."""
    parser = argparse.ArgumentParser(description="Train MNIST SNN model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["DiehlAndCook2015", "IncreasingInhibitionNetwork"],
        default=None,
        help="Model architecture to use (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Batch size for training",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        dest="n_epochs",
        help="Number of training epochs",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        dest="n_train",
        help="Number of training samples (for quick testing, default from config)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        dest="gpu",
        default=None,
        help="Use GPU if available",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        dest="checkpoint_dir",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        dest="resume_from",
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        dest="log_dir",
        default="logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        dest="model_name",
        default=None,
        help="Name for the saved model file (without .pt extension). Default: mnist_snn_model",
    )
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        dest="enable_monitoring",
        default=None,
        help="Enable voltage and spike monitors during training (disabled by default for speed)",
    )
    parser.add_argument(
        "--disable-monitoring",
        action="store_false",
        dest="enable_monitoring",
        help="Explicitly disable monitoring during training (default)",
    )
    parser.add_argument(
        "--enable-grad-tracking",
        action="store_true",
        dest="enable_grad_tracking",
        default=None,
        help="Enable gradient tracking (disabled by default, STDP doesn't need it)",
    )
    parser.add_argument(
        "--disable-grad-tracking",
        action="store_false",
        dest="enable_grad_tracking",
        help="Explicitly disable gradient tracking (default)",
    )
    parser.add_argument(
        "--enable-weight-norm",
        action="store_true",
        dest="enable_weight_norm",
        default=None,
        help="Enable weight normalization during training (disabled by default, normalized at update_interval)",
    )
    parser.add_argument(
        "--disable-weight-norm",
        action="store_false",
        dest="enable_weight_norm",
        help="Explicitly disable weight normalization during training (default)",
    )
    
    args = parser.parse_args()
    return train_command(args)


def infer():
    """Inference entry point."""
    parser = argparse.ArgumentParser(description="Run inference with MNIST SNN model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        dest="model_path",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Batch size for inference",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        dest="gpu",
        default=None,
        help="Use GPU if available",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        dest="log_dir",
        default="logs",
        help="Directory for log files",
    )
    
    args = parser.parse_args()
    return infer_command(args)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Neuro-Sim: Production MNIST SNN System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["DiehlAndCook2015", "IncreasingInhibitionNetwork"],
        default=None,
        help="Model architecture to use (overrides config)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Batch size for training",
    )
    train_parser.add_argument(
        "--n-epochs",
        type=int,
        dest="n_epochs",
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--n-train",
        type=int,
        dest="n_train",
        help="Number of training samples (for quick testing, default from config)",
    )
    train_parser.add_argument(
        "--gpu",
        action="store_true",
        dest="gpu",
        help="Use GPU if available",
    )
    train_parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        dest="checkpoint_dir",
        help="Directory for checkpoints",
    )
    train_parser.add_argument(
        "--resume-from",
        type=str,
        dest="resume_from",
        help="Path to checkpoint to resume from",
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        dest="log_dir",
        default="logs",
        help="Directory for log files",
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        dest="model_name",
        default=None,
        help="Name for the saved model file (without .pt extension). Default: mnist_snn_model",
    )
    train_parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        dest="enable_monitoring",
        default=None,
        help="Enable voltage and spike monitors during training (disabled by default for speed)",
    )
    train_parser.add_argument(
        "--disable-monitoring",
        action="store_false",
        dest="enable_monitoring",
        help="Explicitly disable monitoring during training (default)",
    )
    train_parser.add_argument(
        "--enable-grad-tracking",
        action="store_true",
        dest="enable_grad_tracking",
        default=None,
        help="Enable gradient tracking (disabled by default, STDP doesn't need it)",
    )
    train_parser.add_argument(
        "--disable-grad-tracking",
        action="store_false",
        dest="enable_grad_tracking",
        help="Explicitly disable gradient tracking (default)",
    )
    train_parser.add_argument(
        "--enable-weight-norm",
        action="store_true",
        dest="enable_weight_norm",
        default=None,
        help="Enable weight normalization during training (disabled by default, normalized at update_interval)",
    )
    train_parser.add_argument(
        "--disable-weight-norm",
        action="store_false",
        dest="enable_weight_norm",
        help="Explicitly disable weight normalization during training (default)",
    )
    
    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)",
    )
    infer_parser.add_argument(
        "--model-path",
        type=str,
        dest="model_path",
        help="Path to trained model file",
    )
    infer_parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Batch size for inference",
    )
    infer_parser.add_argument(
        "--gpu",
        action="store_true",
        dest="gpu",
        help="Use GPU if available",
    )
    infer_parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU",
    )
    infer_parser.add_argument(
        "--log-dir",
        type=str,
        dest="log_dir",
        default="logs",
        help="Directory for log files",
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        return train_command(args)
    elif args.command == "infer":
        return infer_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

