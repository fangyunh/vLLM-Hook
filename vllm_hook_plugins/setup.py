from setuptools import setup, find_packages

setup(
    name="vllm-hook-plugins",
    version="0.5.11",
    packages=find_packages(),
    # Deliberately UNPINNED on vllm. This branch is validated on 0.21 + the V1 runner
    # and requirement.txt records that known-good env, but 0.23/0.24 are compat-proven
    # and v2_hook (which inherits this file) runs 0.25 — a hard pin here would make
    # those envs uninstallable. _hook_plugin._note_vllm_version() warns at engine init
    # when the running vLLM is off the validated env; the V1 runner is forced regardless.
    install_requires=["vllm", "zstandard"],
    entry_points={
        "vllm.general_plugins": [
            "hook_registry = vllm_hook_plugins:register_plugins",
            "vllm_hook = vllm_hook_plugins._hook_plugin:register",
        ],
    },
    python_requires=">=3.8",
)