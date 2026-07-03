import os
import argparse
from binoculars.cuda_util import check_cuda
from dotenv import load_dotenv
from binoculars.env_utils import doublecheck_env, doublecheck_pkgs

load_dotenv()

# Check and print results
doublecheck_env(".env")  # check environmental variables
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)  # check packages

if check_cuda() == False:
    assert False

_arg_parser = argparse.ArgumentParser(description=__doc__)
_arg_parser.add_argument(
    "--gpu", default="0", help="value for CUDA_VISIBLE_DEVICES (which GPU(s) to use)"
)
_arg_parser.add_argument(
    "--dataset", default="wp", help="which dataset to score (key into DATASETS)"
)
_args, _ = _arg_parser.parse_known_args()

# os.environ.setdefault("CUDA_VISIBLE_DEVICES", _args.gpu)
# os.environ.setdefault("OMP_NUM_THREADS", "16")
# os.environ.setdefault("MKL_NUM_THREADS", "16")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

observer_name = "gpt2"
performer_name = "gpt2"

# observer_name = "tiiuae/Falcon3-1B-Base"
# performer_name = "tiiuae/Falcon3-1B-Instruct"

# observer_name = "tiiuae/falcon-7b"
# performer_name = "tiiuae/falcon-7b-instruct"

from binoculars import Binoculars
bino = Binoculars(observer_name_or_path = observer_name,
                 performer_name_or_path = performer_name)


# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"
sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

print(bino.compute_score(sample_string))  # 0.75661373
print(bino.predict(sample_string))  # 'Most likely AI-Generated'
