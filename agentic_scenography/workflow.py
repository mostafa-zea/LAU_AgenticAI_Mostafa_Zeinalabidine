
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .blender import BlenderRuntime
from .hosted import function_calls


@dataclass(frozen=True)
class ProjectSettings:
    model: str = "gpt-6-astra"
    frames: int = 180
    fps: int = 12
    width: int = 768
    height: int = 512
    samples: int = 32
    max_technical_repairs: int = 2
    max_design_revisions: int = 1
    gpu_backend: str = "AUTO"
    allow_cpu: bool = False
    blender_path: str | None = None

    def __post_init__(self):
        if not 0 <= self.max_design_revisions <= 2:
            raise ValueError("Use zero, one or two design revisions.")
        if not 0 <= self.max_technical_repairs <= 2:
            raise ValueError("Use at most two technical repairs per design iteration.")


class ProjectState(TypedDict, total=False):
    words: list[str]
    intention: dict
    intention_hash: str
    plan: dict
    version: int
    design_revision: int
    technical_repairs: int
    feedback: list[str]
    code: str
    build: dict
    inspection: dict
    preview: dict
    evaluation: dict
    versions: list[dict]
    selected: dict
    final_render: dict
    status: str
    error: str | None
    stopped_reason: str | None


INTERPRETER_PROMPT = """You are the Interpreter agent in an architectural experiment
called Agentic Scenography. Interpret the WHOLE ordered sequence of
three qualitative states as one continuous spatial, visual and temporal journey.
Preserve ambiguity and associative interpretation. Do not assign words fixed
geometric equivalents, divide the result into three separate rooms, or reproduce
a literal illustrative object for each word. Consider overlaps, anticipation,
pacing, contrast and perception. The scene remains abstract, not a finished building.
Write a concise scenographic intention that a Designer can construct and an
Evaluator can compare with rendered evidence. Distinguish physical effects from
artistic representations. Give a brief design rationale, not private reasoning.
Return only JSON with these fields:
intention (120–180 words), rationale (one sentence), states (three objects with
word, interpretation, visible_criteria list), transitions (two short descriptions),
overall_criteria (three concise, observable criteria). Do not write Blender code.
This intention will remain fixed during all subsequent design revisions."""

DESIGNER_PROMPT = """You are the Designer agent. Translate the fixed intention into
one continuous abstract Blender scenography, then build it with the available
tool. Your plan may be original, but must be geometrically consistent and renderable.
The camera travels through the composition over time. Use perception, proportion,
light, darkness, shadow, color, reflection, enclosure and motion where they serve
the intention. Do not use text labels as substitutes for spatial interpretation.
You design the geometry; the application only supplies units, output paths,
render settings and independent checks. Use real tool observations to repair
technical errors. Never claim a render or successful action without its tool result.
Preserve the intention. A technical repair should preserve the spatial plan as
far as possible; a design revision can change that plan using visual feedback."""

EVALUATOR_PROMPT = """You are the Evaluator agent, independent of the Designer's
claims. Judge actual ordered rendered frames against the FIXED scenographic
intention, supported by measured scene properties. First describe visible evidence,
then judge it. Do not infer visible success from persuasive prose or valid code.
Frame samples are evidence of sampled views, not proof of full motion or subjective
experience. Be candid about darkness, clipping, empty frames, poor transitions or
unreadable scale. Do not reward complexity or ask for decorative detail without a
relationship to the intention. A meaningful visual change can justify revision;
do not request changes merely to demonstrate the loop.
Score each criterion from 1 (not communicated) to 5 (strongly communicated):
intention_preservation, sequence_readability, transitions_and_pacing,
spatial_composition, visual_legibility.
Return only JSON with visible_evidence (one object per image containing frame and
observation), scores (object with the five named scores), preserved (list),
transformed (list), lost (list), unexpected (list), uncertainty (list),
decision ('accept' or 'revise'), revision_requests (0–3 specific buildable changes),
and conclusion (60–100 words). You may recommend changes, not rewrite the intention."""

BUILD_TOOL = {
    "type": "function", "name": "build_scene",
    "description": "Construct a new versioned Blender scene from complete Blender Python code. Returns the saved scene or actual execution errors. The controller owns the version and output path. It independently inspects the result before rendering.",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "One brief reason for this action."},
            "blender_code": {"type": "string", "description": "Complete executable Blender Python script, no Markdown fences."},
        },
        "required": ["reason", "blender_code"], "additionalProperties": False,
    },
    "strict": True,
}

SCORE_NAMES = ("intention_preservation", "sequence_readability",
               "transitions_and_pacing", "spatial_composition", "visual_legibility")


def validate_words(words):
    if isinstance(words, str):
        words = [part.strip() for part in re.split(r"\s*(?:→|>|,)\s*", words)]
    if not isinstance(words, (list, tuple)) or len(words) != 3:
        raise ValueError("Enter exactly three qualitative states in their intended order.")
    if any(not isinstance(word, str) or not word.strip() or len(word.strip()) > 80 for word in words):
        raise ValueError("Each state must be a non-empty phrase of at most 80 characters.")
    return [word.strip() for word in words]


def parse_json(text):
    source = text.strip()
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source)
        source = re.sub(r"\s*```$", "", source)
    result = json.loads(source)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")
    return result


def intention_digest(intention):
    return hashlib.sha256(json.dumps(intention, sort_keys=True).encode()).hexdigest()


def inspect_against_plan(inspection, plan):
    """Check only explicitly measurable claims; never score an emotion numerically."""
    result = dict(inspection)
    issues = list(result.get("issues", []))
    objects = {obj["name"]: obj for obj in result.get("objects", [])}
    if sum(obj.get("type") == "MESH" for obj in objects.values()) > 120:
        issues.append("Scene exceeds the prototype limit of 120 mesh objects.")
    measurements = []
    for requirement in plan.get("checkable_requirements", []):
        name, prop = requirement["object"], requirement["property"]
        obj = objects.get(name)
        actual = None if obj is None else obj.get(prop)
        expected = requirement["expected"]
        tolerance = requirement.get("tolerance_m", 0.15)
        passed = (isinstance(actual, (list, tuple)) and len(actual) == 3 and
                  all(abs(float(a) - float(b)) <= tolerance for a, b in zip(actual, expected)))
        measurements.append({"object": name, "property": prop, "expected": expected,
                             "actual": actual, "tolerance_m": tolerance, "passed": passed})
        if not passed:
            issues.append(f"Plan measurement mismatch: {name}.{prop}; expected {expected}, observed {actual}, tolerance {tolerance} m.")
    if result.get("camera_poses"):
        poses = result["camera_poses"]
        if len({tuple(p["location"]) for p in poses}) == 1:
            issues.append("The planned travelling camera is stationary at all sampled frames.")
    result.update(issues=issues, valid=not issues, plan_measurements=measurements)
    return result


class AgenticScenography:
    """One independent run with a fixed intention and bounded revision budgets."""

    def __init__(self, connection, root, settings=None, observer=None,
                 interpreter_prompt=INTERPRETER_PROMPT, designer_prompt=DESIGNER_PROMPT,
                 evaluator_prompt=EVALUATOR_PROMPT):
        self.connection = connection
        self.root = Path(root).resolve()
        self.settings = settings or ProjectSettings()
        s = self.settings
        self.blender = BlenderRuntime(self.root, blender_path=s.blender_path,
            gpu_backend=s.gpu_backend, allow_cpu=s.allow_cpu, frame_count=s.frames,
            fps=s.fps, resolution=(s.width, s.height), samples=s.samples)
        self.observer = observer
        self.prompts = {"Interpreter": interpreter_prompt, "Designer": designer_prompt,
                        "Evaluator": evaluator_prompt}
        self.events = []
        self.model_calls = []
        self.history = []
        self.reference = ""
        self.started = None

    def emit(self, stage, message, **data):
        event = {"stage": stage, "message": message, **data}
        self.events.append(event)
        if self.observer:
            self.observer(event)

    def constraints(self):
        s = self.settings
        return {"units": "metres", "frame_start": 1, "frame_end": s.frames,
                "fps": s.fps, "duration_seconds": s.frames / s.fps,
                "resolution": [s.width, s.height], "samples": s.samples,
                "engine": "Cycles", "max_mesh_objects": 120,
                "design_revisions": s.max_design_revisions,
                "technical_repairs_per_design_iteration": s.max_technical_repairs}

    async def ask(self, role, prompt=None, history=None, tools=None):
        items = history if history is not None else [{"role": "user", "content": prompt}]
        response = await self.connection.respond(model=self.settings.model,
            instructions=self.prompts[role], input_items=items, tools=tools,
            reasoning_effort="medium")
        self.model_calls.append({"role": role, "model": response["model"],
                                 "elapsed": response["elapsed"], "usage": response["usage"]})
        return response

    async def structured(self, role, prompt, validator):
        """One bounded correction for malformed handoffs, separate from design review."""
        history = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            response = await self.ask(role, history=history)
            try:
                value = parse_json(response["text"])
                validator(value)
                return value
            except (ValueError, TypeError, KeyError) as error:
                self.emit("handoff_check", f"{role} handoff needs correction: {error}", raw_response=response["text"])
                if attempt == 1:
                    raise ValueError(f"{role} did not return a valid handoff after one correction: {error}") from None
                history.extend(response["output"])
                history.append({"role": "user", "content": f"Handoff validation failed: {error}. Return corrected JSON only; preserve the task and intention."})

    async def interpret(self, state):
        self.started = time.perf_counter()
        words = validate_words(state["words"])
        self.emit("interpret", "Interpreting " + " → ".join(words))
        template = ChatPromptTemplate.from_template(
            "Interpret this ordered sequence: {sequence}.\nShared constraints: {constraints}")
        prompt = template.format(sequence=" → ".join(words), constraints=json.dumps(self.constraints()))

        def check(value):
            if not isinstance(value.get("intention"), str) or len(value["intention"]) < 80:
                raise ValueError("A substantive written intention is required.")
            if [s["word"].casefold() for s in value.get("states", [])] != [w.casefold() for w in words]:
                raise ValueError("Address the original three states in the original order.")
            if len(value.get("transitions", [])) != 2 or len(value.get("overall_criteria", [])) < 2:
                raise ValueError("Include both transitions and observable overall criteria.")

        intention = await self.structured("Interpreter", prompt, check)
        self.emit("intention", "The reference intention is now fixed.", intention=intention)
        return {"words": words, "intention": intention, "intention_hash": intention_digest(intention),
                "version": 0, "design_revision": 0, "technical_repairs": 0,
                "feedback": [], "versions": [], "status": "running", "error": None}

    async def plan(self, state):
        self.assert_intention(state)
        self.emit("plan", f"Planning design iteration {state['design_revision'] + 1}.")
        prompt = f"""Develop a buildable spatial plan for this fixed intention:
{json.dumps(state['intention'], ensure_ascii=False)}
Constraints: {json.dumps(self.constraints())}
Design feedback from the previous rendered version: {json.dumps(state.get('feedback', []))}
Return JSON only with summary, spatial_strategy, objects (named key elements with
purpose, geometry, dimensions_m, location_m, material), lighting, camera_path
(at least 4 keyframes with frame, position_m, target_m, intended_experience),
timing (how the 3 states overlap and transition), checkable_requirements
(2–5 key STATIC named mesh objects with object, property 'dimensions' or 'location',
expected [x,y,z], tolerance_m), and artistic_approximations.
Use Blender world Z up, metres, frame 1 through {self.settings.frames}. Define all
camera positions and targets numerically. Keep the camera clear of geometry and
make its initial, middle and final views visibly useful. Full geometry must exist
in a single continuous scene. Use strong, readable spatial differences, not just
changing colors. Design for a {self.settings.frames / self.settings.fps:g}-second abstract study; use bounded simple meshes
and efficient lighting. At most 120 mesh objects and no imported assets or fonts.
Measured dimensions mean Blender Object.dimensions at frame 1 (not world-axis
bounding-box extents). Locations are world coordinates at frame 1.
Do not include camera/light objects in checkable_requirements. No code yet."""

        def check(value):
            for key in ("summary", "spatial_strategy", "objects", "lighting", "camera_path", "timing", "checkable_requirements"):
                if not value.get(key):
                    raise ValueError(f"Missing plan field: {key}")
            if len(value["objects"]) > 120:
                raise ValueError("Too many planned objects.")
            path = value["camera_path"]
            if isinstance(path, dict):
                path = path.get("keyframes")
            if not isinstance(path, list) or not all(isinstance(p, dict) for p in path):
                raise ValueError("camera_path must be a list of keyframes, or an object with a keyframes list.")
            if len(path) < 4 or path[0]["frame"] != 1 or path[-1]["frame"] != self.settings.frames:
                raise ValueError("Camera path must cover the whole frame interval with at least four keys.")
            frames = [p["frame"] for p in path]
            if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= self.settings.frames for n in frames):
                raise ValueError("Camera frames must be whole numbers within the animation interval.")
            if frames != sorted(set(frames)):
                raise ValueError("Camera keyframes must increase strictly.")
            for p in path:
                for key in ("position_m", "target_m"):
                    if len(p[key]) != 3 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in p[key]):
                        raise ValueError("Camera vectors must contain three finite numbers.")
                if p["position_m"] == p["target_m"]:
                    raise ValueError("Camera position and target cannot coincide.")
            for item in value["checkable_requirements"]:
                if item["property"] not in ("dimensions", "location") or len(item["expected"]) != 3:
                    raise ValueError("Use supported measurable properties and three coordinates.")
                if not all(isinstance(n, (int, float)) and math.isfinite(n) for n in item["expected"]):
                    raise ValueError("Expected measurements must be three finite numbers.")
                tolerance = item.get("tolerance_m", 0.15)
                if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance <= 0.5:
                    raise ValueError("Measurement tolerance must be >0 and ≤0.5 metres.")

        plan = await self.structured("Designer", prompt, check)
        self.history = []
        self.emit("spatial_plan", "Spatial plan passed the handoff checks.", plan=plan)
        return {"plan": plan, "error": None}

    async def build(self, state):
        self.assert_intention(state)
        version = state["version"] + 1
        if not self.blender.startup:
            startup = await self.blender.start()
            self.emit("blender", "Dedicated Blender session started.", startup=startup)
            self.reference = await self.blender.api_reference()
        if not self.history:
            self.history = [{"role": "user", "content": self.build_instruction(state)}]
        else:
            self.history.append({"role": "user", "content":
                "The technical gate failed. Repair the full scene code using this observation, "
                "preserving the intention and spatial plan. Request build_scene once.\n" + str(state.get("error"))})
        self.emit("build", f"Designer is generating scene version {version}.")
        response = await self.ask("Designer", history=self.history, tools=[BUILD_TOOL])
        self.history.extend(response["output"])
        try:
            calls = function_calls(response)
            if len(calls) != 1 or calls[0]["name"].split(".")[-1] != "build_scene":
                raise ValueError("Designer must request exactly one build_scene action.")
            call = calls[0]
            code = call["arguments"]["blender_code"]
            reason = call["arguments"].get("reason", response["text"])
            self.emit("tool_request", reason, tool="build_scene", version=version, code=code)
            observation = await self.blender.build_scene(code, version)
            self.history.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(observation)})
            self.emit("tool_observation", "Blender returned its execution result.", tool="build_scene", observation=observation)
            error = None if observation.get("status") == "ok" else observation.get("message", json.dumps(observation))
            return {"version": version, "code": code, "build": observation, "error": error}
        except (ValueError, KeyError, TypeError) as error:
            # No invalid or ambiguous tool request is executed.
            self.history = []
            return {"version": version, "error": str(error), "build": {"status": "error", "message": str(error)}}

    def build_instruction(self, state):
        return f"""Use build_scene ONCE with a complete Python script for this plan.
Fixed intention: {json.dumps(state['intention'], ensure_ascii=False)}
Spatial plan: {json.dumps(state['plan'], ensure_ascii=False)}
Technical feedback, if any: {state.get('error')}
Constraints: {json.dumps(self.constraints())}
The application provides a dedicated EMPTY active scene and world. Use
scene = bpy.context.scene. It owns units, Cycles GPU selection, resolution, frame
range, output paths, saving and rendering. Do not reset the application, switch
scenes, clear unrelated data, save or render. Build only in the supplied scene.
Imports allowed: bpy, math, mathutils, random (no aliases; Vector import allowed).
No OS, network, files, subprocess, eval/exec, getattr/setattr, dunder names,
bpy.app, preferences, drivers, handlers or while loops. Do not set file paths.
No external fonts/assets, text labels, imported models, or enormous subdivided meshes.
Use documented Blender 5.1 API; procedural mesh/material/light/camera code only.
Always create a camera and set scene.camera. Animate location and rotation with
keyframe_insert; at every planned key compute rotation from target-position using
Vector(...).to_track_quat('-Z','Y').to_euler(). Camera local -Z looks forward.
Use cube size=1 with dimensions directly (not division by 2), update and apply
scale before Boolean operations. Give key meshes EXACTLY the planned names.
Material node type identifiers include ShaderNodeBsdfPrincipled,
ShaderNodeEmission, ShaderNodeOutputMaterial, ShaderNodeVolumeScatter,
ShaderNodeOutputWorld. Lights use bpy.data.lights.new or bpy.ops.object.light_add.
For emission prefer Principled inputs 'Emission Color' and 'Emission Strength'.
Use nodes links explicitly. Avoid compositor configuration unless essential;
Blender 5.1 compositor API changed. Avoid broad exception suppression.
For camera keyframes, leave default interpolation if unsure of Blender5.1 action API.
Provide readable lighting and useful silhouettes. Avoid unbounded world fog;
if justified, use a finite, low-density volume. Ensure no enclosing opaque object
blocks all camera views. Control owns rendering, not design.
Official API excerpts retrieved through the Blender MCP documentation tool:
{self.reference[:20000]}
Request build_scene with full code and a brief action reason. Do not merely print code."""

    async def inspect(self, state):
        self.emit("inspect", f"Checking scene version {state['version']} against technical requirements.")
        observation = await self.blender.inspect_scene(state["version"])
        if observation.get("status") == "ok":
            details = inspect_against_plan(observation["result"], state["plan"])
            observation["result"] = details
            error = None if details["valid"] else "\n".join(details["issues"])
        else:
            error = observation.get("message", "Scene inspection failed.")
        self.emit("technical_checks", "Independent measurements returned.", inspection=observation)
        return {"inspection": observation, "error": error}

    async def preview(self, state):
        frames = sorted({1, self.settings.frames // 4, self.settings.frames // 2,
                         3 * self.settings.frames // 4, self.settings.frames})
        self.emit("preview", f"Rendering five ordered views of version {state['version']}.")
        observation = await self.blender.render_sequence(state["version"], frames=frames)
        error = None if observation.get("status") == "ok" else observation.get("message", "Preview rendering failed.")
        self.emit("preview_result", "Preview render result.", preview=observation)
        return {"preview": observation, "error": error}

    async def evaluate(self, state):
        self.assert_intention(state)
        self.emit("evaluate", "Evaluator is comparing actual image pixels with the fixed intention.")
        content = [{"type": "input_text", "text":
            "Fixed intention: " + json.dumps(state["intention"], ensure_ascii=False) +
            "\nIndependent measurements: " + json.dumps(state["inspection"]["result"]) +
            "\nEvaluate the following ordered frames. The intended run lasts " +
            str(self.settings.frames / self.settings.fps) + " seconds. Give the required JSON."}]
        preview = state["preview"]["result"]
        for frame, path in zip(preview["frames"], preview["frame_paths"]):
            content.extend([
                {"type": "input_text", "text": f"Frame {frame}, time {(frame - 1) / self.settings.fps:.2f} seconds"},
                {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()},
            ])
        history = [{"role": "user", "content": content}]
        for attempt in range(2):
            response = await self.ask("Evaluator", history=history)
            try:
                evaluation = parse_json(response["text"])
                scores = evaluation["scores"]
                if set(scores) != set(SCORE_NAMES) or not all(isinstance(n, (int, float)) and not isinstance(n, bool) and 1 <= n <= 5 for n in scores.values()):
                    raise ValueError("All five rubric scores must be numbers from 1 to 5.")
                if evaluation["decision"] not in ("accept", "revise"):
                    raise ValueError("Decision must be accept or revise.")
                if not isinstance(evaluation.get("revision_requests"), list):
                    raise ValueError("revision_requests must be a list.")
                break
            except (ValueError, TypeError, KeyError) as error:
                self.emit("evaluation_handoff", str(error), raw_response=response["text"])
                if attempt:
                    raise ValueError("Evaluator handoff failed after correction.") from None
                history.extend(response["output"])
                history.append({"role": "user", "content": "Correct the JSON handoff: " + str(error)})
        evaluation["mean_score"] = round(sum(evaluation["scores"].values()) / len(SCORE_NAMES), 2)
        record = {"version": state["version"], "design_revision": state["design_revision"],
                  "plan": state["plan"], "build": state["build"], "inspection": state["inspection"],
                  "preview": state["preview"], "evaluation": evaluation}
        self.emit("design_review", "Design review completed; technical validity alone did not determine this judgment.", evaluation=evaluation)
        return {"evaluation": evaluation, "versions": state["versions"] + [record], "error": None}

    def technical_route(self, state):
        if not state.get("error"):
            return "continue"
        if state["technical_repairs"] < self.settings.max_technical_repairs:
            return "repair"
        return "select" if state.get("versions") else "stop"

    async def repair(self, state):
        count = state["technical_repairs"] + 1
        self.emit("technical_repair", f"Technical repair {count}/{self.settings.max_technical_repairs}; intention and plan stay fixed.", error=state["error"])
        return {"technical_repairs": count}

    def design_route(self, state):
        if state["evaluation"]["decision"] == "revise" and state["design_revision"] < self.settings.max_design_revisions:
            return "revise"
        return "select"

    async def revise(self, state):
        number = state["design_revision"] + 1
        evaluation = state["evaluation"]
        feedback = evaluation["revision_requests"] + ["Previous conclusion: " + evaluation["conclusion"]]
        self.emit("design_revision", f"Design revision {number}/{self.settings.max_design_revisions}; returning to spatial planning, keeping the intention unchanged.", feedback=feedback)
        return {"design_revision": number, "technical_repairs": 0, "feedback": feedback, "error": None}

    async def select(self, state):
        candidates = state.get("versions", [])
        if not candidates:
            return {"status": "failed", "stopped_reason": state.get("error", "No valid evaluated version exists.")}
        selected = max(candidates, key=lambda v: (v["evaluation"]["mean_score"], -v["version"]))
        reason = state.get("error") or ("Design revision limit reached with unresolved feedback."
            if state["evaluation"]["decision"] == "revise" else None)
        self.emit("selection", f"Selected technically valid version {selected['version']} by the recorded rubric scores; ties favor the earlier version.", selected_version=selected["version"], unresolved=reason)
        return {"selected": selected, "stopped_reason": reason, "error": None}

    async def finish(self, state):
        selected = state.get("selected")
        if not selected:
            return {"status": "failed"}
        self.emit("final_render", f"Rendering the full {self.settings.frames}-frame animation from version {selected['version']}.")
        observation = None
        for attempt in range(self.settings.max_technical_repairs + 1):
            observation = await self.blender.render_sequence(selected["version"], final=True)
            if observation.get("status") == "ok":
                break
            self.emit("export_repair", "Export did not complete; retrying the same selected scene within the technical limit.", attempt=attempt, observation=observation)
        passed = observation.get("status") == "ok"
        self.emit("final_output", "Animation export complete." if passed else "Animation export failed; evidence is retained.", final_render=observation)
        return {"final_render": observation, "status": "complete" if passed else "failed",
                "error": None if passed else observation.get("message", "Export failed.")}

    async def stop(self, state):
        self.emit("stopped", "Technical repair budget exhausted; no valid evaluated scene is available.", error=state.get("error"))
        return {"status": "failed", "stopped_reason": state.get("error")}

    def assert_intention(self, state):
        if intention_digest(state["intention"]) != state["intention_hash"]:
            raise ValueError("The fixed intention changed; refusing to continue.")

    def summary(self, state):
        usage = [call["usage"] for call in self.model_calls]
        return {"status": state.get("status"), "sequence": state["words"],
                "model": self.settings.model, "settings": asdict(self.settings),
                "model_calls": len(self.model_calls),
                "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
                "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
                "model_seconds": round(sum(c["elapsed"] for c in self.model_calls), 2),
                "elapsed_seconds": round(time.perf_counter() - self.started, 2) if self.started else None,
                "scene_build_attempts": state.get("version", 0),
                "design_revisions": state.get("design_revision", 0),
                "evaluated_versions": len(state.get("versions", [])),
                "selected_version": state.get("selected", {}).get("version"),
                "unresolved": state.get("stopped_reason"),
                "user_review": "The complete animation still requires the author's viewing and judgment."}


def build_graph(project):
    """Connect the stages and their repair and revision routes."""
    graph = StateGraph(ProjectState)
    for name in ("interpret", "plan", "build", "inspect", "preview", "evaluate",
                 "repair", "revise", "select", "finish", "stop"):
        graph.add_node(name, getattr(project, name))
    graph.add_edge(START, "interpret")
    graph.add_edge("interpret", "plan")
    graph.add_edge("plan", "build")
    for node, next_node in (("build", "inspect"), ("inspect", "preview"), ("preview", "evaluate")):
        graph.add_conditional_edges(node, project.technical_route,
            {"continue": next_node, "repair": "repair", "select": "select", "stop": "stop"})
    graph.add_edge("repair", "build")
    graph.add_conditional_edges("evaluate", project.design_route, {"revise": "revise", "select": "select"})
    graph.add_edge("revise", "plan")
    graph.add_edge("select", "finish")
    graph.add_edge("finish", END)
    graph.add_edge("stop", END)
    return graph.compile()
