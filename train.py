from Tools import load_config
import argparse
from trainer import Trainer
from Restore.render_prediction import render_prediction

def main(config):
    trainer = Trainer(config)
    if 'train' in config['arguments']['task']:
        trainer.train()
    if 'test' in config['arguments']['task']:
        trainer.test()
        save_images = trainer.save_path_results / f"test_{trainer.ckpt['epoch']}_images"
        save_images.mkdir()
        render_prediction(trainer, save_images).render()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task", nargs="+")
    arg_parser.add_argument("--config", "-c")
    arg_parser.add_argument("--resume-from", "-r")
    arg_parser.add_argument("--test-set", "-t")
    arg_parser.add_argument("--image-path", "-i")
    arg_parser.add_argument("--num-workers", type=int)
    arguments = arg_parser.parse_args()
    config = load_config(arguments.config, arguments)
    main(config)