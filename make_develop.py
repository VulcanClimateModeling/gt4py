#!/usr/bin/env python3

import argparse
import configparser
import enum
import os.path
import re
import subprocess
import sys


class ColorPrint:
    @staticmethod
    def failure(message, end="\n"):
        sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def good(message, end="\n"):
        sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def warn(message, end="\n"):
        sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def info(message, end="\n"):
        sys.stdout.write("\x1b[1;34m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def bold(message, end="\n"):
        sys.stdout.write("\x1b[1;37m" + message.strip() + "\x1b[0m" + end)


class ExitCode(enum.IntEnum):
    NONEXISTANT_SECTION = 1
    NONEXISTANT_KEY = 2
    NOT_ANCESTOR = 3
    MERGE_CONFLICT = 4
    WRITE_ERROR = 5


def exit_failure(message, code=1):
    ColorPrint.failure(message)
    exit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", help="Overwrite directory", action="store_true")
    parser.add_argument("--git", help="Custom git executable", type=str, default="git", nargs=1)
    parser.add_argument("directory", help="Directory to build branch in")
    parser.add_argument("config", help="Configuration file")
    parse_args = parser.parse_args()

    script_name = sys.argv[0]
    directory = parse_args.directory
    config_file = parse_args.config
    git_exe = parse_args.git

    directory_existed = os.path.exists(directory)

    if directory_existed and not parse_args.force:
        exit_failure(
            "Directory already exists. Use --force or -f to overwrite directory. Exiting...",
            code=ExitCode.WRITE_ERROR,
        )

    config = configparser.ConfigParser()
    config.read(config_file)

    if "Base" not in config.sections():
        exit_failure('Requires "Base" section in config file', code=ExitCode.NONEXISTANT_SECTION)

    def get_git_object(section, remote):
        if "branch" in section:
            return remote + "/" + section["branch"]
        elif "hash" in section:
            return section["hash"]
        else:
            exit_failure(
                f'Feature sections require either "branch" or "hash" key',
                code=ExitCode.NONEXISTANT_KEY,
            )

    for section in config.sections():
        get_git_object(config[section], "origin")

    def run_git_cmd(*args, cwd=directory):
        print("RUNNING", git_exe, *args)
        return subprocess.run([git_exe, *args], cwd=cwd, stdout=subprocess.PIPE)

    # Clone
    base_section = config["Base"]
    run_git_cmd("clone", base_section["url"], directory, cwd="./")

    # Get final branch name
    final_branch = base_section.get("final", None)
    if not final_branch:
        exit_failure(
            '"Base" section is missing "final" branch name', code=ExitCode.NONEXISTANT_KEY
        )

    # Checkout base git object
    base_git_object = get_git_object(base_section, "origin")
    run_git_cmd("checkout", "-b", final_branch, base_git_object)

    # Loop over features
    features = set(config.sections()) - {"Base"}
    for feature in features:
        section = config[feature]

        git_config = configparser.ConfigParser()
        git_config.read(os.path.join(directory, ".git/config"))
        remote_urls = {
            section.split()[1].strip('"'): git_config[section]["url"]
            for section in filter(lambda key: "remote" in key, git_config.sections())
        }
        remote_name = None
        for remote, url in remote_urls.items():
            if url == section["url"].strip():
                remote_name = remote
                break

        if remote_name is None:
            remote_name = re.sub(r"\s+", "", feature)
            run_git_cmd("remote", "add", remote_name, section["url"])
            run_git_cmd("fetch", remote_name)

        git_object = get_git_object(section, remote_name)

        # # Check if the base branch is an ancestor of this
        # if run_git_cmd("merge-base", "--is-ancestor", base_git_object, git_object).returncode != 0:
        #     exit_failure("Not an ancestor", code=ExitCode.NOT_ANCESTOR)

        strategy_args = []
        if "strategy" in section.keys():
            merge_strategy = section["strategy"]
            strategy_args = ["-s", merge_strategy]

        # Merge it in
        run_git_cmd("merge", "-m", f"Merged {git_object}", *strategy_args, git_object)

        # Check for conflicts
        if run_git_cmd("ls-files", "-u").stdout:
            ColorPrint.warn(
                "Merge conflict detected. Fix the issues, `git add`, then press ENTER to continue..."
            )
            user_input = sys.stdin.readline().rstrip()
            if user_input == "cancel" or user_input == "exit":
                exit_failure("Exiting...", code=ExitCode.MERGE_CONFLICT)
            else:
                run_git_cmd("commit", "-m", f"Merged {git_object}")

    if "upstream" in base_section.keys():
        run_git_cmd("remote", "add", "upstream", base_section["upstream"])

    print(f"Adding {script_name} and {config_file}")
    script_in_directory = os.path.join(directory, os.path.split(script_name)[-1])
    config_in_directory = os.path.join(directory, os.path.split(config_file)[-1])
    subprocess.run(["cp", script_name, script_in_directory])
    subprocess.run(["cp", config_file, config_in_directory])
    run_git_cmd(
        "add", os.path.split(script_in_directory)[-1], os.path.split(config_in_directory)[-1]
    )
    run_git_cmd("commit", "-m", f"Added {script_in_directory} and {config_in_directory}")

    print(f"Now run:\n\t$ cd {directory}\n\t$ git push -f upstream {final_branch}")

