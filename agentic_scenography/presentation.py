
from io import BytesIO
from pathlib import Path

from IPython.display import Image, Markdown, Video, display
from PIL import Image as PILImage, ImageDraw


def contact_sheet(paths, labels=None):
    paths = list(paths)
    if not paths:
        return
    width, height = 384, 256
    columns = min(3, len(paths))
    rows = (len(paths) + columns - 1) // columns
    sheet = PILImage.new("RGB", (columns * width, rows * (height + 28)), "#eeeeec")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        with PILImage.open(path) as source:
            picture = source.convert("RGB")
            picture.thumbnail((width, height))
        x, y = (i % columns) * width, (i // columns) * (height + 28)
        sheet.paste(picture, (x + (width - picture.width) // 2, y))
        label = labels[i] if labels else Path(path).name
        draw.text((x + 8, y + height + 6), str(label), fill="#202020")
    buffer = BytesIO()
    sheet.save(buffer, format="PNG")
    display(Image(data=buffer.getvalue()))


def notebook_observer(event):
    print(f"\n{event['stage'].replace('_', ' ').capitalize()}: {event['message']}")
    if "intention" in event:
        display(Markdown(event["intention"]["intention"]))
    if "plan" in event:
        display(Markdown(event["plan"]["summary"]))
    if "startup" in event:
        info = event["startup"]
        print(f"Blender {info.get('blender_version')} | {info.get('engine')} | {info.get('backend')} {info.get('device')} | metres")
    if "observation" in event:
        observation = event["observation"]
        print("Tool status:", observation.get("status"))
        if observation.get("status") != "ok":
            print(str(observation.get("message", observation))[-1800:])
    if "inspection" in event:
        observation = event["inspection"]
        result = observation.get("result", {})
        passed = observation.get("status") == "ok" and result.get("valid", False)
        print("Technical checks:", "passed" if passed else "failed")
        measurements = result.get("plan_measurements", [])
        if measurements:
            print("Planned measurements matched:", sum(bool(m.get("passed")) for m in measurements), "/", len(measurements))
        for item in result.get("issues", []) + result.get("warnings", []):
            print("-", item)
        if observation.get("status") != "ok":
            print(str(observation.get("message", observation))[-1800:])
    if "preview" in event:
        observation = event["preview"]
        if observation.get("status") == "ok":
            result = observation["result"]
            contact_sheet(result["frame_paths"], [f"Frame {n}" for n in result["frames"]])
        else:
            print(str(observation.get("message", observation))[-1800:])
    if "evaluation" in event:
        review = event["evaluation"]
        display(Markdown(review["conclusion"]))
        print("Scores / 5:", "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in review["scores"].items()))
        for key in ("preserved", "transformed", "lost", "unexpected", "uncertainty", "revision_requests"):
            if review.get(key):
                print("\n" + key.replace("_", " ").capitalize() + ":")
                for item in review[key]:
                    print("-", item)
        print("\nDesign-review decision:", review["decision"])
    if event.get("error"):
        print(str(event["error"])[-1800:])
    if event.get("unresolved"):
        print("Unresolved:", event["unresolved"])
    if "final_render" in event:
        result = event["final_render"]
        print("Export status:", result.get("status"))
        if result.get("status") == "ok":
            output = result["result"]
            print("Frames:", len(output.get("frame_paths", [])), "| Seconds:", output.get("duration_seconds"))
        else:
            print(str(result.get("message", result))[-1800:])


def show_result(project, state):
    summary = project.summary(state)
    print("\nRun result:", summary.get("status"), "| Selected version:", summary.get("selected_version"))
    print("Build attempts:", summary.get("scene_build_attempts"), "| Design revisions:", summary.get("design_revisions"), "| Model calls:", summary.get("model_calls"))
    if state.get("final_render", {}).get("status") == "ok":
        output = state["final_render"]["result"]
        display(Video(filename=output["video_path"], embed=True, width=768))
        print("Watch the complete movement and pacing before judging the relation to the intention.")
    if summary.get("unresolved"):
        print("Remaining limitation:", summary["unresolved"])
    return summary
