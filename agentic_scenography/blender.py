
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import timedelta
from pathlib import Path


MCP_COMMIT = "dbbf836ad4b1025f14a2b3b504c43903f39e0b04"
MCP_REPOSITORY = "https://projects.blender.org/lab/blender_mcp.git"


def normalize_mcp(value):
    """Unwrap LangChain/MCP text blocks, preserving Blender's error envelope."""
    if isinstance(value, dict):
        if "status" in value or "found" in value or "hits" in value:
            return value
        if "content" in value:
            return normalize_mcp(value["content"])
        return value
    if isinstance(value, str):
        try:
            return normalize_mcp(json.loads(value))
        except json.JSONDecodeError:
            return {"status": "error", "message": value}
    if isinstance(value, list):
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text":
                return normalize_mcp(block.get("text", ""))
        return {"status": "error", "message": "The MCP tool returned no text result."}
    return {"status": "error", "message": f"Unexpected MCP result type: {type(value).__name__}"}


def _attribute_path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def validate_blender_code(code):
    """Reject syntax errors and obvious file, OS, network or process operations.

    This deliberately remains a small, explainable guard, not an assertion that
    arbitrary Python can be made safe by parsing its syntax tree.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Blender code must be a non-empty Python string.")
    if len(code) > 100_000:
        raise ValueError("Generated code exceeds the 100,000-character prototype limit.")
    tree = ast.parse(code, filename="model_generated_scene.py")
    permitted_imports = {"bpy", "math", "mathutils", "random"}
    blocked_names = {
        "eval", "exec", "compile", "open", "input", "breakpoint", "exit", "quit",
        "globals", "locals", "vars", "getattr", "setattr", "delattr", "__import__",
        "memoryview", "help",
    }
    blocked_attributes = {
        "filepath", "filepath_raw", "save", "save_render", "save_as_mainfile", "save_mainfile",
        "open_mainfile", "read_homefile", "read_factory_settings", "user_preferences",
        "preferences", "handlers", "timers", "driver_namespace", "driver_add", "load",
        "write", "write_text", "write_bytes", "unlink", "remove_file", "system", "popen",
    }
    safe_operator_groups = {"mesh", "curve", "object", "transform", "anim", "paint"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in permitted_imports or alias.asname:
                    raise ValueError("Only unaliased bpy, math, mathutils and random imports are allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in permitted_imports or any(a.name == "*" for a in node.names):
                raise ValueError("Only explicit imports from bpy, math, mathutils and random are allowed.")
        elif isinstance(node, ast.Name) and (node.id in blocked_names or node.id.startswith("__")):
            raise ValueError(f"Disallowed Python name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in blocked_attributes:
                raise ValueError(f"Disallowed attribute: {node.attr}")
            path = _attribute_path(node)
            if path.startswith("bpy.app") or path.startswith("bpy.utils") or path.startswith("bpy.path"):
                raise ValueError(f"Generated scene code cannot access {path}.")
            if path.startswith("bpy.ops."):
                group = path.split(".")[2]
                if group not in safe_operator_groups:
                    raise ValueError(f"Operator group bpy.ops.{group} is controlled by the runtime.")
        elif isinstance(node, (ast.While, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
            raise ValueError("Unbounded while loops and asynchronous code are not allowed in scene scripts.")
    return tree


def _find_blender(explicit=None):
    candidates = [explicit, os.environ.get("BLENDER_EXECUTABLE"), shutil.which("blender")]
    if sys.platform == "darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    elif sys.platform == "win32":
        candidates.extend(str(p) for p in sorted(Path("C:/Program Files/Blender Foundation").glob("Blender */blender.exe"), reverse=True))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("Install Blender 5.1.2, then set blender_path or BLENDER_EXECUTABLE to its executable.")


def _find_mcp_source(explicit=None):
    candidates = [explicit, os.environ.get("SCENOGRAPHY_MCP_SOURCE"),
                  Path(__file__).resolve().parents[1] / "blender" / "blender_mcp"]
    spec = importlib.util.find_spec("blmcp")
    if spec and spec.origin:
        candidates.append(Path(spec.origin).resolve().parents[2])
    for candidate in candidates:
        if candidate:
            source = Path(candidate).expanduser().resolve()
            if (source / "addon" / "blender_mcp_addon").is_dir() and (source / "mcp" / "blmcp").is_dir():
                return source
    raise FileNotFoundError("BlenderLab MCP source is missing. Run the notebook's dependency setup cell first.")


def _version_number(version):
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Version must be a positive integer.")
    return version


class BlenderRuntime:
    """Own one visible Blender process and versioned output folder per project run."""

    def __init__(self, root, blender_path=None, gpu_backend="AUTO", allow_cpu=False,
                 frame_count=180, fps=12, resolution=(768, 512), samples=32,
                 mcp_source=None, render_timeout=1800):
        self.root = Path(root).expanduser().resolve()
        self.blender_path = blender_path
        self.gpu_backend = str(gpu_backend).upper()
        self.allow_cpu = bool(allow_cpu)
        self.frame_count = int(frame_count)
        self.fps = int(fps)
        self.resolution = tuple(int(n) for n in resolution)
        self.samples = int(samples)
        if not 2 <= self.frame_count <= 600 or not 1 <= self.fps <= 60:
            raise ValueError("Prototype bounds: 2–600 frames and 1–60 fps.")
        if len(self.resolution) != 2 or any(n < 64 or n > 2048 or n % 2 for n in self.resolution):
            raise ValueError("Resolution must contain two even pixel counts between 64 and 2048.")
        if not 1 <= self.samples <= 256:
            raise ValueError("Prototype bounds: 1–256 Cycles samples.")
        self.mcp_source = mcp_source
        self.render_timeout = float(render_timeout)
        self.process = None
        self.client = None
        self.tools = {}
        self.startup = None
        self._temporary = None
        self._versions = {}
        self._attempts = {}
        self._reference = None
        self._operation_lock = asyncio.Lock()

    async def start(self):
        """Launch factory-startup GUI Blender; never attach to the user's other scene."""
        if self.process is not None and self.process.poll() is None:
            return self.startup
        from langchain_mcp_adapters.client import MultiServerMCPClient
        self.root.mkdir(parents=True, exist_ok=True)
        executable = _find_blender(self.blender_path)
        source = _find_mcp_source(self.mcp_source)
        self.mcp_source = source
        self._temporary = tempfile.TemporaryDirectory(prefix="scenography-blender-")
        report_path = Path(self._temporary.name) / "startup.json"
        # Bind first to select a currently free loopback port. Blender will bind it
        # after launch; a competing process can still take it, reported as failure.
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        env = os.environ.copy()
        env.update(SCENOGRAPHY_BOOT_REPORT=str(report_path), SCENOGRAPHY_MCP_SOURCE=str(source),
                   SCENOGRAPHY_MCP_PORT=str(port), SCENOGRAPHY_GPU_BACKEND=self.gpu_backend,
                   SCENOGRAPHY_ALLOW_CPU="1" if self.allow_cpu else "0")
        self.process = subprocess.Popen(
            [executable, "--factory-startup", "--python", str(Path(__file__).with_name("blender_bootstrap.py"))],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 90
        while not report_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"Blender closed during startup (exit {self.process.returncode}).")
            if time.monotonic() >= deadline:
                raise TimeoutError("Blender did not finish startup within 90 seconds. Check its window.")
            await asyncio.sleep(0.5)
        # The bootstrap writes a small file in one operation; retry if read during
        # the brief creation/write interval.
        for attempt in range(10):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                if attempt == 9:
                    raise
                await asyncio.sleep(0.1)
        if report.get("status") != "ready":
            raise RuntimeError(report.get("error", "Blender startup failed."))
        self.startup = report
        self.client = MultiServerMCPClient({"blender": {
            # The same official blmcp.main entry point as `python -m blmcp`;
            # silence repetitive INFO request logs, retaining warnings/errors.
            "transport": "stdio", "command": sys.executable,
            "args": ["-c", "import logging; logging.disable(logging.INFO); import blmcp; raise SystemExit(blmcp.main())"],
            "env": {"BLENDER_MCP_HOST": "127.0.0.1", "BLENDER_MCP_PORT": str(port),
                    "PYTHONPATH": str(source / "mcp")},
            "session_kwargs": {"read_timeout_seconds": timedelta(seconds=310)},
        }})
        self.tools = {tool.name: tool for tool in await self.client.get_tools()}
        if "execute_blender_code" not in self.tools:
            raise RuntimeError("The connected Blender MCP server does not expose execute_blender_code.")
        return report

    async def execute(self, code):
        """Execute trusted infrastructure code; generated code uses build_scene."""
        if not self.tools or self.process is None or self.process.poll() is not None:
            raise RuntimeError("Start the Blender runtime before using its tools.")
        response = await self.tools["execute_blender_code"].ainvoke({"code": code})
        return normalize_mcp(response)

    def _settings_code(self):
        device = "CPU" if self.startup and self.startup.get("backend") == "CPU" else "GPU"
        return textwrap.dedent(f"""
            scene.unit_settings.system = 'METRIC'
            scene.unit_settings.length_unit = 'METERS'
            scene.unit_settings.scale_length = 1.0
            scene.render.engine = 'CYCLES'
            scene.cycles.device = {device!r}
            scene.cycles.samples = {self.samples}
            scene.cycles.use_denoising = True
            scene.render.resolution_x = {self.resolution[0]}
            scene.render.resolution_y = {self.resolution[1]}
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'RGB'
            scene.render.fps = {self.fps}
            scene.render.fps_base = 1.0
            scene.frame_start = 1
            scene.frame_end = {self.frame_count}
            scene.frame_step = 1
        """)

    @staticmethod
    def _unload_scenes_code():
        # Blender object/material names are global across scenes. Keeping old
        # versions loaded would silently suffix new names with .001, breaking
        # checks against the plan. Saved versions remain immutable on disk and
        # are appended back when selected for rendering.
        return textwrap.dedent("""
            import bpy
            if bpy.app.is_job_running('RENDER'):
                raise RuntimeError('A render is running; wait before changing the active scene.')
            setup_scene = bpy.data.scenes.get('Scenography_Setup')
            if setup_scene is None:
                setup_scene = bpy.data.scenes.new('Scenography_Setup')
            bpy.context.window.scene = setup_scene
            for old_scene in list(bpy.data.scenes):
                if old_scene.name.startswith('Scenography_v'):
                    for old_object in list(old_scene.objects):
                        bpy.data.objects.remove(old_object, do_unlink=True)
                    bpy.data.scenes.remove(old_scene)
            bpy.data.orphans_purge(do_recursive=True)
        """)

    async def _ensure_version_loaded(self, version):
        saved = self._get_version(version)
        code = f"import bpy\nresult = {{'loaded': {saved['scene_name']!r} in bpy.data.scenes}}\n"
        response = await self.execute(code)
        if response.get("status") != "ok" or response["result"]["loaded"]:
            return response
        code = self._unload_scenes_code()
        code += f"\nwith bpy.data.libraries.load({saved['scene_path']!r}, link=False) as (available, requested):\n    requested.scenes = [{saved['scene_name']!r}]\n"
        code += f"scene = bpy.data.scenes[{saved['scene_name']!r}]\nbpy.context.window.scene = scene\nresult = {{'loaded': True, 'scene_name': scene.name}}\n"
        return await self.execute(code)

    async def build_scene(self, blender_code, version):
        """Build unedited model code in a fresh scene and preserve its source.

        A failed attempt remains available as an attempt .py file. Successful
        versions are immutable; design revision must use a new version number.
        """
        version = _version_number(version)
        async with self._operation_lock:
            destination = self.root / f"version_{version:02d}"
            destination.mkdir(parents=True, exist_ok=True)
            blend_path = destination / f"scene_v{version:02d}.blend"
            if version in self._versions or blend_path.exists():
                return {"status": "error", "message": "This version already has a saved scene. Use a new version number."}
            attempt = len(list(destination.glob("attempt_*.py"))) + 1
            code_path = destination / f"attempt_{attempt:02d}.py"
            code_path.write_text(blender_code, encoding="utf-8")
            try:
                validate_blender_code(blender_code)
            except (SyntaxError, ValueError) as error:
                return {"status": "error", "stage": "code_validation", "message": str(error), "code_path": str(code_path)}
            scene_name = f"Scenography_v{version:02d}_attempt{attempt:02d}"
            prelude = self._unload_scenes_code()
            prelude += f"scene = bpy.data.scenes.new({scene_name!r})\nbpy.context.window.scene = scene\nscene.world = bpy.data.worlds.new({(scene_name + '_World')!r})\nscene.world.use_nodes = True\n"
            source = prelude + self._settings_code() + "\n" + blender_code + "\n"
            # Re-select our scene rather than trust a generated reassignment of
            # scene or context. The renderer belongs to the controller.
            source += f"\nscene = bpy.data.scenes[{scene_name!r}]\nbpy.context.window.scene = scene\n"
            source += self._settings_code()
            source += "\nscene.frame_set(1)\nbpy.context.view_layer.update()\n"
            source += textwrap.dedent("""
                import math
                if scene.camera is None or scene.camera not in list(scene.objects):
                    raise RuntimeError('Technical precondition: the scene needs an active camera before it can be saved as a successful build.')
                if not any(item.type == 'MESH' for item in scene.objects):
                    raise RuntimeError('Technical precondition: the scene contains no mesh geometry.')
                for item in scene.objects:
                    if not all(math.isfinite(float(v)) for v in list(item.location) + list(item.dimensions)):
                        raise RuntimeError('Technical precondition: non-finite transform on ' + item.name)
            """)
            source += f"\nbpy.ops.wm.save_as_mainfile(filepath={str(blend_path)!r}, check_existing=False)\n"
            source += f"result = {{'scene_name': scene.name, 'scene_path': {str(blend_path)!r}, 'code_path': {str(code_path)!r}, 'version': {version}, 'attempt': {attempt}, 'object_count': len(scene.objects)}}\n"
            response = await self.execute(source)
            response["code_path"] = str(code_path)
            if response.get("status") == "ok":
                self._versions[version] = {"scene_name": scene_name, "scene_path": str(blend_path), "code_path": str(code_path)}
            return response

    def _get_version(self, version):
        version = _version_number(version)
        if version not in self._versions:
            raise ValueError("Build this scene version successfully in the current runtime before inspecting or rendering it.")
        return self._versions[version]

    async def inspect_scene(self, version):
        """Measure the built scene independently of the model's prose claims."""
        saved = self._get_version(version)
        loaded = await self._ensure_version_loaded(version)
        if loaded.get("status") != "ok":
            return loaded
        code = "import bpy, math\nif bpy.app.is_job_running('RENDER'):\n    raise RuntimeError('A render is running; wait before inspection changes the sampled frame.')\n"
        code += f"scene = bpy.data.scenes[{saved['scene_name']!r}]\nbpy.context.window.scene = scene\n"
        code += textwrap.dedent("""
            issues = []
            warnings = []
            object_records = []
            for obj in scene.objects:
                values = list(obj.location) + list(obj.dimensions) + list(obj.rotation_euler)
                if not all(math.isfinite(float(value)) for value in values):
                    issues.append('Non-finite transform: ' + obj.name)
                if obj.type == 'MESH' and not len(obj.data.vertices):
                    issues.append('Empty mesh: ' + obj.name)
                object_records.append({
                    'name': obj.name, 'type': obj.type,
                    'dimensions': [float(v) for v in obj.dimensions],
                    'location': [float(v) for v in obj.matrix_world.translation],
                    'local_location': [float(v) for v in obj.location],
                    'world_location': [float(v) for v in obj.matrix_world.translation],
                    'materials': [m.name if m else None for m in getattr(obj.data, 'materials', [])],
                    'hidden_in_render': bool(obj.hide_render),
                    'animated': bool(obj.animation_data),
                })
            if scene.camera is None or scene.camera not in list(scene.objects):
                issues.append('No active camera in this scene.')
            if not any(o.type == 'MESH' for o in scene.objects):
                issues.append('Scene contains no mesh geometry.')
            if scene.unit_settings.system != 'METRIC' or scene.unit_settings.scale_length != 1:
                issues.append('Scene is not configured at one Blender unit per metre.')
            lights = [{'name': o.name, 'type': o.data.type, 'energy': o.data.energy}
                      for o in scene.objects if o.type == 'LIGHT']
            if not lights:
                warnings.append('No light objects; check whether world or emissive materials provide sufficient illumination.')
            camera_poses = []
            previous_frame = scene.frame_current
            from mathutils import Vector
            if scene.camera:
                for frame in sorted({scene.frame_start, (scene.frame_start + scene.frame_end) // 2, scene.frame_end}):
                    scene.frame_set(frame)
                    bpy.context.view_layer.update()
                    camera = scene.camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
                    location = camera.matrix_world.translation
                    direction = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
                    if not all(math.isfinite(float(v)) for v in list(location) + list(direction)):
                        issues.append('Non-finite camera pose at frame ' + str(frame))
                    camera_poses.append({'frame': frame, 'location': list(location), 'view_direction': list(direction),
                                         'lens_mm': camera.data.lens, 'clip_start': camera.data.clip_start,
                                         'clip_end': camera.data.clip_end})
                if len({tuple(p['location']) + tuple(p['view_direction']) for p in camera_poses}) == 1:
                    warnings.append('Camera has the same pose at sampled frames; verify whether geometry or lighting supplies the intended temporal change.')
            scene.frame_set(previous_frame)
            result = {
                'valid': not issues, 'issues': issues, 'warnings': warnings,
                'scene_name': scene.name, 'camera': scene.camera.name if scene.camera else None,
                'engine': scene.render.engine, 'render_device': scene.cycles.device,
                'backend': bpy.app.driver_namespace.get('scenography_backend'),
                'units': scene.unit_settings.length_unit, 'metres_per_unit': scene.unit_settings.scale_length,
                'frame_range': [scene.frame_start, scene.frame_end], 'fps': scene.render.fps,
                'objects': object_records, 'lights': lights, 'camera_poses': camera_poses,
                'camera_constraints': [{'name': c.name, 'type': c.type, 'enabled': not c.mute}
                                       for c in scene.camera.constraints] if scene.camera else [],
                'limitation': 'Technical validity does not establish spatial or experiential success; rendered evidence is evaluated separately.'
            }
        """)
        return await self.execute(code)

    async def api_reference(self):
        """Retrieve bounded official API excerpts through MCP, with local fallback.

        This is documentation retrieval, not model training. Sources and query
        status are included in the returned text for notebook provenance.
        """
        if self._reference is not None:
            return self._reference
        topics = [
            "bpy.types.Object.dimensions", "bpy.types.BooleanModifier",
            "bpy.types.ShaderNodeEmission", "bpy.types.ShaderNodeVolumeScatter",
            "bpy.types.CameraDOFSettings", "mathutils.Vector.to_track_quat",
            "bpy.types.bpy_struct.keyframe_insert",
        ]
        blocks = ["Official Blender API excerpts retrieved from the pinned BlenderLab MCP documentation bundle.\n"
                  "Reference: https://docs.blender.org/api/5.1/\n"
                  "MCP source: " + MCP_REPOSITORY + " @ " + MCP_COMMIT]
        for topic in topics:
            content = ""
            method = "MCP exact lookup"
            try:
                tool = self.tools.get("get_python_api_docs")
                if tool:
                    document = normalize_mcp(await tool.ainvoke({"identifier": topic}))
                    if document.get("found"):
                        content = document.get("content", "")
                if not content and self.tools.get("search_api_docs"):
                    method = "MCP search fallback"
                    parent, member = topic.rsplit(".", 1)
                    found = normalize_mcp(await self.tools["search_api_docs"].ainvoke({
                        "query": f"{parent} {member}", "max_results": 2, "context": 4,
                    }))
                    content = "\n".join(str(item.get("text", "")) for item in found.get("hits", []))
            except Exception as error:
                method = f"Local pinned documentation fallback after {type(error).__name__}"
            if not content and self.mcp_source:
                content = _local_api_excerpt(Path(self.mcp_source), topic)
                method = "Local pinned documentation fallback"
            if content:
                blocks.append(f"\n{topic} ({method})\n{content[:4500]}")
            else:
                blocks.append(f"\n{topic}: documentation unavailable; do not treat the missing excerpt as verified API guidance.")
        self._reference = "\n".join(blocks)
        return self._reference

    async def _wait_for_render(self, expected_paths, timeout):
        deadline = time.monotonic() + timeout
        # INVOKE_DEFAULT schedules a job; allow its first tick before polling.
        await asyncio.sleep(1.0)
        while True:
            if self.process.poll() is not None:
                return {"status": "error", "message": "Blender closed while rendering."}
            response = await self.execute("import bpy\nresult = {'rendering': bpy.app.is_job_running('RENDER')}\n")
            if response.get("status") != "ok":
                return response
            if not response["result"]["rendering"]:
                missing = [str(path) for path in expected_paths if not _valid_png(path)]
                if missing:
                    return {"status": "error", "message": "Rendering stopped before all images were saved. Rerun to resume.", "missing_frames": missing}
                return {"status": "ok", "result": {"image_count": len(expected_paths)}}
            if time.monotonic() >= deadline:
                return {"status": "error", "message": "Render wait timed out. Blender may still be rendering; inspect its window before another render request."}
            await asyncio.sleep(2.0)

    async def render_sequence(self, version, frames=None, final=False):
        """Render preview samples or the final PNG sequence and encode an MP4.

        Valid existing images from an immutable version are reused. A final
        animation can be resumed after interruption, without rerendering saved
        frames. Rendering uses Blender's GUI job, avoiding long MCP calls.
        """
        saved = self._get_version(version)
        async with self._operation_lock:
            inspection = await self.inspect_scene(version)
            if inspection.get("status") != "ok" or not inspection.get("result", {}).get("valid"):
                return {"status": "error", "message": "Rendering requires a technically valid saved scene.", "inspection": inspection}
            requested = list(range(1, self.frame_count + 1)) if final else sorted(set(frames or [1, self.frame_count // 2, self.frame_count]))
            if not requested or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= self.frame_count for n in requested):
                raise ValueError(f"Frames must be integers between 1 and {self.frame_count}.")
            folder = self.root / f"version_{version:02d}" / ("animation_frames" if final else "previews")
            folder.mkdir(parents=True, exist_ok=True)
            paths = [folder / f"frame_{frame:04d}.png" for frame in requested]
            for path in paths:
                if path.exists() and not _valid_png(path):
                    path.unlink()  # Only incomplete PNGs in this run's owned output folder.
            missing = [(frame, path) for frame, path in zip(requested, paths) if not path.exists()]
            started = time.perf_counter()
            setup = f"import bpy\nscene = bpy.data.scenes[{saved['scene_name']!r}]\nbpy.context.window.scene = scene\n"
            setup += self._settings_code()
            setup += "\nif bpy.app.is_job_running('RENDER'):\n    raise RuntimeError('A render is already running. Wait for it to finish.')\n"
            if missing and final:
                code = setup + f"\nscene.render.filepath = {str(folder / 'frame_')!r}\nscene.render.use_file_extension = True\nscene.render.use_overwrite = False\nscene.render.use_placeholder = False\n"
                code += "operation = bpy.ops.render.render('INVOKE_DEFAULT', animation=True)\nresult = {'started': 'RUNNING_MODAL' in operation, 'operator_result': list(operation)}\n"
                response = await self.execute(code)
                if response.get("status") != "ok":
                    return response
                response = await self._wait_for_render(paths, self.render_timeout)
                if response.get("status") != "ok":
                    return response
            elif missing:
                for frame, path in missing:
                    code = setup + f"\nscene.frame_set({frame})\nscene.render.filepath = {str(path)!r}\nscene.render.use_overwrite = True\n"
                    code += "operation = bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)\nresult = {'started': 'RUNNING_MODAL' in operation, 'operator_result': list(operation)}\n"
                    response = await self.execute(code)
                    if response.get("status") != "ok":
                        return response
                    response = await self._wait_for_render([path], self.render_timeout)
                    if response.get("status") != "ok":
                        return response
            result = {"version": version, "frames": requested, "frame_paths": [str(p) for p in paths],
                      "paths": [str(p) for p in paths], "final": bool(final), "backend": self.startup["backend"],
                      "resolution": list(self.resolution), "samples": self.samples,
                      "seconds": round(time.perf_counter() - started, 2), "reused_images": len(paths) - len(missing)}
            if final:
                result["video_path"] = str(await asyncio.to_thread(self._encode_video, folder))
                result["duration_seconds"] = self.frame_count / self.fps
            return {"status": "ok", "result": result}

    def _encode_video(self, folder):
        import imageio_ffmpeg
        output = folder.parent / "scenography.mp4"
        temporary = folder.parent / "scenography.encoding.mp4"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-framerate", str(self.fps), "-start_number", "1", "-i", str(folder / "frame_%04d.png"),
                   "-frames:v", str(self.frame_count), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-crf", "18", "-movflags", "+faststart", str(temporary)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError("Video encoding failed: " + completed.stderr[-2000:])
        os.replace(temporary, output)
        return output

    async def close(self):
        """Close only the dedicated Blender process owned by this instance."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, timeout=5)
        self.tools = {}
        self.client = None
        if self._temporary:
            self._temporary.cleanup()


def _valid_png(path):
    if not Path(path).is_file():
        return False
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def _local_api_excerpt(source, topic):
    folder = source / "mcp" / "blmcp" / "data" / "api"
    parts = topic.split(".")
    for split in range(len(parts), 0, -1):
        path = folder / (".".join(parts[:split]) + ".rst")
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            if split == len(parts):
                return content[:4500]
            member = parts[-1]
            lines = content.splitlines()
            for index, line in enumerate(lines):
                if (".. " in line or "def " in line) and member in line:
                    return "\n".join(lines[max(0, index - 1):index + 38])
    return ""
