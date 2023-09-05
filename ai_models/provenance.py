# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
import sysconfig


def version(versions, name, module, roots, namespaces):
    try:
        versions[name] = module.__version__
        return
    except AttributeError:
        pass

    try:
        path = module.__file__

        if path is None:
            namespaces.add(name)
            return

        for k, v in roots.items():
            path = path.replace(k, f"<{v}>")

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

    versions = {}
    namespaces = set()
    for k, v in sorted(sys.modules.items()):
        if "." not in k:
            version(versions, k, v, roots, namespaces)

    # Catter for modules like "earthkit.meteo"
    for k, v in sorted(sys.modules.items()):
        bits = k.split(".")
        if len(bits) == 2 and bits[0] in namespaces:
            version(versions, k, v, roots, namespaces)

    return versions


def platform_info():
    import platform

    return dict(
        system=platform.system(),
        release=platform.release(),
        version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
    )


def gather_provenance_info():
    executable = sys.executable

    return dict(
        executable=executable,
        python_path=sys.path,
        config_paths=sysconfig.get_paths(),
        modules=module_versions(),
        platform=platform_info(),
    )
