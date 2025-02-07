# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess  # nosec
import sys

from openvino.runtime import get_version as get_ie_version

from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.find_ie_version import find_ie_version
from openvino.tools.mo.utils.utils import get_mo_root_dir
from openvino.tools.ovc.version import extract_release_version, simplify_version, extract_hash_from_version, \
    SingletonMetaClass  # pylint: disable=no-name-in-module,import-error


def get_version_file_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "version.txt")


def generate_mo_version():
    """
    Function generates version like in cmake
    custom_{branch_name}_{commit_hash}
    """
    try:
        mo_dir = get_mo_root_dir()
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=mo_dir).strip().decode()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mo_dir).strip().decode()
        return "custom_{}_{}".format(branch_name, commit_hash)
    except Exception as e:
        return "unknown version"


def get_version():
    version_txt = get_version_file_path()
    if not os.path.isfile(version_txt):
        return generate_mo_version()
    with open(version_txt) as f:
        return f.readline().replace('\n', '')


def get_simplified_mo_version():
    return simplify_version(get_version())


def get_simplified_ie_version(env=dict(), version=None):
    if version is None:
        try:
            version = subprocess.check_output([sys.executable, os.path.join(os.path.dirname(__file__), "ie_version.py")], timeout=2, env=env).strip().decode()
        except:
            return "ie not found"

    # To support legacy IE versions
    m = re.match(r"^([0-9]+).([0-9]+).(.*)", version)
    if m and len(m.groups()) == 3:
        return simplify_version(m.group(3))
    return simplify_version(version)


class VersionChecker(metaclass=SingletonMetaClass):
    def __init__(self):
        self.runtime_checked = False
        self.mo_version = None
        self.ie_version = None
        self.mo_simplified_version = None
        self.ie_simplified_version = None

    def get_mo_version(self):
        if self.mo_version:
            return self.mo_version
        self.mo_version = get_version()
        return self.mo_version

    def get_ie_version(self):
        if self.ie_version:
            return self.ie_version
        self.ie_version = get_ie_version()
        return self.ie_version

    def get_mo_simplified_version(self):
        if self.mo_simplified_version:
            return self.mo_simplified_version
        self.mo_simplified_version = simplify_version(self.get_mo_version())
        return self.mo_simplified_version

    def get_ie_simplified_version(self):
        if self.ie_simplified_version:
            return self.ie_simplified_version
        self.ie_simplified_version = get_simplified_ie_version(env=os.environ)
        return self.ie_simplified_version

    def check_runtime_dependencies(self, silent=True):
        if not self.runtime_checked:
            def raise_ie_not_found():
                raise Error("Could not find the Inference Engine or nGraph Python API.\n"
                            "Consider building the Inference Engine and nGraph Python APIs from sources or "
                            "try to install OpenVINO (TM) Toolkit using pip \npip install openvino")

            try:
                if not find_ie_version(silent=silent):
                    raise_ie_not_found()
            except Exception as e:
                import logging as log
                if log is not None:
                    log.error(e)
                raise_ie_not_found()
            self.runtime_checked = True
