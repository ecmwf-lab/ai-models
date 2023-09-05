# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import os
import sys
import sysconfig


def lookup_git_repo(path):
    from git import InvalidGitRepositoryError, Repo

    while path != "/":
        try:
            return Repo(path)
        except InvalidGitRepositoryError:
            path = os.path.dirname(path)

    return None


def check_for_git(paths):
    versions = {}
    for name, path in paths:
        repo = lookup_git_repo(path)
        if repo is None:
            continue

        versions[name] = dict(
            path=path,
            git=dict(
                sha1=repo.head.commit.hexsha,
                remotes=[r.url for r in repo.remotes],
                modified_files=sorted([item.a_path for item in repo.index.diff(None)]),
                untracked_files=sorted(repo.untracked_files),
            ),
        )

    return versions


def version(versions, name, module, roots, namespaces, paths):
    path = None

    if hasattr(module, "__file__"):
        path = module.__file__
        if path is not None:
            for k, v in roots.items():
                path = path.replace(k, f"<{v}>")

            if path.startswith("/"):
                paths.add((name, path))

    try:
        versions[name] = module.__version__
        return
    except AttributeError:
        pass

    try:
        if path is None:
            namespaces.add(name)
            return

        # For now, don't report on stdlib modules
        if path.startswith("<stdlib>"):
            return

        versions[name] = path
        return
    except AttributeError:
        pass

    if name in sys.builtin_module_names:
        return

    versions[name] = str(module)


def module_versions():
    # https://docs.python.org/3/library/sysconfig.html

    roots = {}
    for name, path in sysconfig.get_paths().items():
        if path not in roots:
            roots[path] = name

    # Sort by length of path, so that we get the most specific first
    roots = {
        path: name
        for path, name in sorted(roots.items(), key=lambda x: len(x[0]), reverse=True)
    }

    paths = set()

    versions = {}
    namespaces = set()
    for k, v in sorted(sys.modules.items()):
        if "." not in k:
            version(versions, k, v, roots, namespaces, paths)

    # Catter for modules like "earthkit.meteo"
    for k, v in sorted(sys.modules.items()):
        bits = k.split(".")
        if len(bits) == 2 and bits[0] in namespaces:
            version(versions, k, v, roots, namespaces, paths)

    git_versions = check_for_git(paths)

    return versions, git_versions


def platform_info():
    import platform

    r = {}
    for p in dir(platform):
        if p.startswith("_"):
            continue
        try:
            r[p] = getattr(platform, p)()
        except Exception:
            pass

    def all_empty(x):
        return all(all_empty(v) if isinstance(v, (list, tuple)) else v == "" for v in x)

    for k, v in list(r.items()):
        if isinstance(v, (list, tuple)) and all_empty(v):
            del r[k]

    return r


def gpu_info():
    import nvsmi

    if not nvsmi.is_nvidia_smi_on_path():
        return "nvdia-smi not found"

    return [json.loads(gpu.to_json()) for gpu in nvsmi.get_gpus()]


def path_md5(path):
    import hashlib

    hash = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash.update(chunk)
    return hash.hexdigest()


def assets_info(paths):
    result = {}

    for path in paths:
        try:
            (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(path)
            md5 = path_md5(path)
        except Exception as e:
            result[path] = str(e)
            continue

        result[path] = dict(
            size=size,
            atime=datetime.datetime.fromtimestamp(atime).isoformat(),
            mtime=datetime.datetime.fromtimestamp(mtime).isoformat(),
            ctime=datetime.datetime.fromtimestamp(ctime).isoformat(),
            md5=md5,
        )

        try:
            from .checkpoint import peek

            result[path]["peek"] = peek(path)
        except Exception:
            pass

    return result


def gather_provenance_info(assets):
    executable = sys.executable

    versions, git_versions = module_versions()

    return dict(
        time=datetime.datetime.utcnow().isoformat(),
        executable=executable,
        args=sys.argv,
        python_path=sys.path,
        config_paths=sysconfig.get_paths(),
        module_versions=versions,
        git_versions=git_versions,
        platform=platform_info(),
        gpus=gpu_info(),
        assets=assets_info(assets),
    )
