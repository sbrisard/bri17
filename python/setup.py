import configparser
import os.path

import pybind11
import setuptools


def get_metadata(key):
    with open(os.path.join("..", "metadata", key + ".txt"), "r", encoding="utf8") as f:
        return f.read().strip()


if __name__ == "__main__":
    metadata = {
        "name": "pybri17",
        "version": get_metadata("version"),
        "author": get_metadata("author"),
        "author_email": "email",
        "description": get_metadata("description"),
        "url": get_metadata("repository"),
    }

    with open(os.path.join("..", "README.md"), "r", encoding="utf8") as f:
        metadata["long_description"] = f.read()

    config = configparser.ConfigParser()
    config.read("setup.cfg")
    bri17_include_dir = config["bri17"].get("include_dir", "")

    pybri17 = setuptools.Extension(
        "pybri17",
        include_dirs=[pybind11.get_include(),
                      bri17_include_dir],
        sources=["pybri17.cpp"],
        extra_compile_args=["-D__DOC__=\"\"{}\"\"".format(metadata["description"]),
                            "-D__AUTHOR__=\"\"{}\"\"".format(metadata["author"]),
                            "-D__VERSION__=\"\\\"{}\\\"\"".format(metadata["version"]), "/std:c++latest"]
    )

    setuptools.setup(
        long_description_content_type="text/markdown",
        ext_modules=[pybri17],
        **metadata
    )
