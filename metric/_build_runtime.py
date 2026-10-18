# Copyright 2026 Mirza Husadzic. SPDX-License-Identifier: GPL-3.0-only
"""Package the maintained RoSE scorer, leaving upstream experimental variants out."""
from setuptools.command.build_py import build_py


class RuntimeBuild(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == "rose":
            modules = [item for item in modules if item[1] in {"__init__", "main", "scorer_hash"}]
        return modules
