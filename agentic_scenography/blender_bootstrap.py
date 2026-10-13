import json
import os
import sys
import traceback
from pathlib import Path

import bpy


def bootstrap():
    report_path = Path(os.environ["SCENOGRAPHY_BOOT_REPORT"])
    report = {"status": "starting", "blender_version": bpy.app.version_string}
    try:
        source = Path(os.environ["SCENOGRAPHY_MCP_SOURCE"])
        sys.path.insert(0, str(source / "addon"))
        from blender_mcp_addon import execute_interactive, mcp_to_blender_server

        if bpy.app.version < (5, 1, 0):
            raise RuntimeError("This project requires Blender 5.1 or newer; validated on 5.1.2.")
        scene = bpy.data.scenes.new("Scenography_Setup")
        bpy.context.window.scene = scene
        # This is a newly launched factory-startup process owned by the project.
        # Remove its default objects so their global Blender names cannot collide
        # with the model's explicitly named objects in the first version.
        for old_scene in list(bpy.data.scenes):
            if old_scene != scene:
                for obj in list(old_scene.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.scenes.remove(old_scene)
        bpy.data.orphans_purge(do_recursive=True)
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.length_unit = "METERS"
        scene.unit_settings.scale_length = 1.0
        scene.render.engine = "CYCLES"

        preferences = bpy.context.preferences.addons["cycles"].preferences
        requested = os.environ.get("SCENOGRAPHY_GPU_BACKEND", "AUTO")
        if requested == "AUTO":
            backends = ["METAL"] if sys.platform == "darwin" else ["OPTIX", "CUDA", "HIP", "ONEAPI"]
        else:
            backends = [requested]
        selected = None
        attempts = []
        for backend in backends:
            try:
                preferences.compute_device_type = backend
                devices = preferences.get_devices_for_type(backend)
                if any(device.type == backend for device in devices):
                    selected = backend
                    break
            except Exception as error:
                attempts.append({"backend": backend, "error": str(error)})
        if selected:
            for device in preferences.devices:
                device.use = device.type == selected
            scene.cycles.device = "GPU"
            if selected == "METAL" and hasattr(preferences, "metalrt"):
                preferences.metalrt = "ON"
        elif os.environ.get("SCENOGRAPHY_ALLOW_CPU") == "1":
            scene.cycles.device = "CPU"
            selected = "CPU"
        else:
            raise RuntimeError(
                f"No supported {requested} GPU was found. Explicitly set allow_cpu=True "
                f"only if you accept CPU rendering. Backend checks: {attempts}"
            )
        bpy.app.driver_namespace["scenography_backend"] = selected
        port = int(os.environ["SCENOGRAPHY_MCP_PORT"])
        mcp_to_blender_server.start("127.0.0.1", port)
        bpy.app.timers.register(execute_interactive.run, first_interval=0.25, persistent=True)
        report.update(
            status="ready", pid=os.getpid(), host="127.0.0.1", port=port,
            scene=scene.name, engine=scene.render.engine,
            device=scene.cycles.device, backend=selected,
            units=scene.unit_settings.length_unit,
            devices=[{"name": item.name, "type": item.type, "enabled": bool(item.use)}
                     for item in preferences.devices],
            metal_ray_tracing=getattr(preferences, "metalrt", None),
        )
    except Exception as error:
        report.update(status="error", error=str(error), traceback=traceback.format_exc())
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


bootstrap()
